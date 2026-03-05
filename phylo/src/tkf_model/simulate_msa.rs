use std::cell::RefCell;

use hashbrown::HashMap;
use log::warn;
use rand::{Rng, RngCore, SeedableRng};
use rand_distr::{Distribution, Geometric};

use crate::alignment::{AncestralAlignment, Sequences, MASA};
use crate::alphabets::AMB_CHAR;
use crate::phylo_info::PhyloInfo;
use crate::random::RandomGenerator;
use crate::substitution_models::{QMatrix, SubstModel};
use crate::tkf_model::{beta, h1, n0, TKFModel};
use crate::tree::{NodeIdx, Tree};
use crate::{record_wo_desc as record, Result};

/// Abstracts over how a single fragment's length is sampled.
///
/// TKF92 draws from a geometric distribution parameterised by `r`.
/// TKF91 always returns length 1 (each residue is its own independent link).
///
/// Returns `(length, log_probability)` so that the caller can accumulate
/// the simulation log-likelihood without any model-specific logic.
pub trait FragmentSampler {
    fn sample_fragment_length<R: Rng>(&self, rng: &mut R) -> (usize, f64);
}

const DELETION_CHAR: u8 = b'-';
const FRAGMENT_BOUNDARY_CHAR: u8 = b',';
const NOTHING_CHAR: u8 = b'_';

/// Since these sequences are built incrementally, we can't use the [`Sequences`] which hold immutable [`record`]s`.
type Seqs = HashMap<NodeIdx, Vec<u8>>;

pub struct TKFMSASimulator<
    T: TKFModel + FragmentSampler,
    Q: QMatrix,
    R: Rng + SeedableRng + RngCore,
> {
    indel_model: T,
    _subst_model: SubstModel<Q>,
    tree: Tree,
    cumulative_logl: RefCell<f64>,
    rng: RefCell<RandomGenerator<R>>,
    max_insertion_length: usize,
}

/// All the descendant links associated to a single branch.
type BranchLinkChildren = Vec<TKFLink>;

struct TKFLink {
    /// The in the tree the link is associated with.
    node: NodeIdx,
    /// True if the link is immortal. It's the link to the very left of the sequence.
    is_immortal: bool,
    /// How many characters are associated to the right of this link.
    length: usize,
    /// For every child node/branch of this link's node, the descendant links.
    children: Vec<BranchLinkChildren>,
    /// For every child node/branch of this link's node, the [`LinkFate`] of this link.
    fates: Vec<LinkFate>,
    is_insertion: bool,
}

impl TKFLink {
    fn new_immortal(node: NodeIdx) -> Self {
        TKFLink {
            node,
            is_immortal: true,
            length: 0,
            children: Vec::new(),
            fates: Vec::new(),
            is_insertion: false,
        }
    }
    fn new(node: NodeIdx, length: usize) -> Self {
        TKFLink {
            node,
            is_immortal: false,
            length,
            children: Vec::new(),
            fates: Vec::new(),
            is_insertion: false,
        }
    }

    fn new_insertion(node: NodeIdx, length: usize) -> Self {
        TKFLink {
            node,
            is_immortal: false,
            length,
            children: Vec::new(),
            fates: Vec::new(),
            is_insertion: true,
        }
    }
}

#[derive(Clone)]
enum LinkFate {
    /// The link is deleted.
    Deletion,
    /// The link survives on the branch and produces `usize` many insertions to the right of it.
    Homolog(usize),
    /// The link is deleted but produces `usize` many insertions to the right of it.
    NonHomolog(usize),
}

pub struct TKF92MSASimulationResult {
    msa: MASA,
    msa_with_non_emitting_cols: MASA,
    fragmentation: Vec<usize>,
    logl: f64,
}

impl TKF92MSASimulationResult {
    pub fn msa(&self) -> &MASA {
        &self.msa
    }

    pub fn msa_with_non_emitting_cols(&self) -> &MASA {
        &self.msa_with_non_emitting_cols
    }

    pub fn fragmentation(&self) -> &Vec<usize> {
        &self.fragmentation
    }

    pub fn logl(&self) -> f64 {
        self.logl
    }
}

impl<T: TKFModel + FragmentSampler, Q: QMatrix, R: Rng + SeedableRng + RngCore>
    TKFMSASimulator<T, Q, R>
{
    pub fn new(
        indel_model: T,
        substitution_model: SubstModel<Q>,
        tree: Tree,
        rng: RandomGenerator<R>,
        max_insertion_length: usize,
    ) -> Self {
        Self {
            indel_model,
            _subst_model: substitution_model,
            tree,
            cumulative_logl: RefCell::new(0.0),
            rng: RefCell::new(rng),
            max_insertion_length,
        }
    }

    pub(crate) fn _double_check_simulation_logl_with_cost_calculation(
        &self,
        result: &TKF92MSASimulationResult,
    ) -> Result<()> {
        let _phylo = PhyloInfo {
            msa: result.msa.clone(),
            tree: self.tree.clone(),
        };
        // TODO: wait until the tkf tree seach cost pr is merged, then i can use that here
        Ok(())
    }

    fn sample_tkf_link_fate(&self, time: f64) -> LinkFate {
        let uniform_sample = self.rng.borrow_mut().random::<f64>();
        let lambda = self.indel_model.lambda();
        let mu = self.indel_model.mu();
        let beta = beta(lambda, mu, time);
        let n_0 = n0(self.indel_model.mu(), beta);
        let homolog_prob_integrated = homolog_prob_integrated(mu, time);
        let non_homolog_prob_integrated = non_homolog_prob_integrated(mu, beta, time);
        debug_assert!(
            (n_0 + homolog_prob_integrated + non_homolog_prob_integrated - 1.0).abs() < 1e-10
        );
        // Link is deleted without producing any insertions
        if uniform_sample < n_0 {
            *self.cumulative_logl.borrow_mut() += (n_0).ln();
            LinkFate::Deletion
        } else if uniform_sample < n_0 + homolog_prob_integrated {
            // Link survives homologously
            let adjusted_sample = uniform_sample - n_0;
            self.sample_homolog_fate(time, adjusted_sample)
        } else {
            // Link is deleted but has non-homologous insertions
            debug_assert!(1.0 - uniform_sample < non_homolog_prob_integrated);
            let adjusted_sample = uniform_sample - n_0 - homolog_prob_integrated;
            self.sample_non_homolog_fate(time, adjusted_sample)
        }
    }

    fn sample_homolog_fate(&self, time: f64, uniform_sample: f64) -> LinkFate {
        let lambda = self.indel_model.lambda();
        let mu = self.indel_model.mu();
        let beta = beta(lambda, mu, time);
        let mut cumulative_prob = 0.0;
        for n in 1..self.max_insertion_length + 1 {
            let homolog_prob = homolog_prob(n, lambda, mu, beta, time);
            cumulative_prob += homolog_prob;
            if uniform_sample < cumulative_prob {
                *self.cumulative_logl.borrow_mut() += homolog_prob.ln();
                return LinkFate::Homolog(n - 1); // n - 1 because we are not counting the original link
            }
        }
        // didnt return in the loop, capping insertion at length max_insertion_length
        let homolog_prob_integrated = homolog_prob_integrated(mu, time);
        let homolog_prob = homolog_prob_integrated - cumulative_prob;
        *self.cumulative_logl.borrow_mut() += homolog_prob.ln();
        warn!(
            "Capping homologous insertion length at {}",
            self.max_insertion_length
        );
        LinkFate::Homolog(self.max_insertion_length)
    }

    fn sample_non_homolog_fate(&self, time: f64, uniform_sample: f64) -> LinkFate {
        let lambda = self.indel_model.lambda();
        let mu = self.indel_model.mu();
        let beta = beta(lambda, mu, time);
        let mut cumulative_prob = 0.0;
        for n in 1..self.max_insertion_length {
            let non_homolog_prob = non_homolog_prob(n, lambda, mu, beta, time);
            cumulative_prob += non_homolog_prob;
            if uniform_sample < cumulative_prob {
                *self.cumulative_logl.borrow_mut() += non_homolog_prob.ln();
                return LinkFate::NonHomolog(n);
            }
        }
        // didnt return ind the loop, capping insertion at length max_insertion_length
        let non_homolog_prob_integrated = non_homolog_prob_integrated(mu, beta, time);
        let non_homolog_prob = non_homolog_prob_integrated - cumulative_prob;
        *self.cumulative_logl.borrow_mut() += non_homolog_prob.ln();
        warn!(
            "Capping non-homologous insertion length at {}",
            self.max_insertion_length
        );
        LinkFate::NonHomolog(self.max_insertion_length)
    }

    // TODO: this is geometric and can be sampled directly
    // However, then I should also cap the max number of insertions
    fn sample_tkf_immortal_link_fate(&self, time: f64) -> usize {
        let uniform_sample = self.rng.borrow_mut().random::<f64>();
        let lambda = self.indel_model.lambda();
        let beta = beta(lambda, self.indel_model.mu(), time);
        let mut comulative_prob = 0.0;
        for n in 1..self.max_insertion_length {
            let immortal_prob = immortal_prob(n, lambda, beta);
            comulative_prob += immortal_prob;
            if uniform_sample < comulative_prob {
                *self.cumulative_logl.borrow_mut() += immortal_prob.ln();
                return n - 1;
            }
        }
        let immortal_prob = 1.0 - comulative_prob;
        *self.cumulative_logl.borrow_mut() += immortal_prob.ln();
        self.max_insertion_length
    }

    fn sample_num_root_links(&self) -> usize {
        let prob_of_success = 1.0 - self.indel_model.lambda() / self.indel_model.mu(); // ie stopping the links
        let geom = Geometric::new(prob_of_success).unwrap();
        let choice = geom.sample(&mut self.rng.borrow_mut().rng);
        let prob = (1.0 - prob_of_success).powi(choice as i32) * prob_of_success;
        *self.cumulative_logl.borrow_mut() += prob.ln();
        choice as usize
    }

    fn sample_fragment_length(&self) -> usize {
        let (length, log_prob) = self
            .indel_model
            .sample_fragment_length(&mut self.rng.borrow_mut().rng);
        *self.cumulative_logl.borrow_mut() += log_prob;
        length
    }

    fn build_root_links(&self) -> Vec<TKFLink> {
        let num_root_links = self.sample_num_root_links();
        let mut root_links = Vec::with_capacity(num_root_links + 1); // +1 for the immortal link
        root_links.push(TKFLink::new_immortal(self.tree.root));
        for _ in 0..num_root_links {
            let length = self.sample_fragment_length();
            root_links.push(TKFLink::new(self.tree.root, length));
        }
        root_links
    }

    fn evolve_insertions(&self, number: usize, parent_link: &mut TKFLink, branch_id: usize) {
        for _ in 0..number {
            let length = self.sample_fragment_length();
            let node = self.tree.children(&parent_link.node)[branch_id];
            let insertion_link = TKFLink::new_insertion(node, length);
            parent_link.children[branch_id].push(insertion_link);
            self.evolve_link_down_tree(parent_link.children[branch_id].last_mut().unwrap());
        }
    }

    fn evolve_link_down_tree(&self, link: &mut TKFLink) {
        for (branch_id, child_node) in self.tree.children(&link.node).iter().enumerate() {
            link.children.push(Vec::new());
            let branch_length = self.tree.node(child_node).blen;
            if link.is_immortal {
                //  immortal link always survives and evolves down the tree
                let child_link = TKFLink::new_immortal(*child_node);
                link.children[branch_id].push(child_link);
                self.evolve_link_down_tree(link.children[branch_id].last_mut().unwrap());
                // new insertions
                let num_insertions = self.sample_tkf_immortal_link_fate(branch_length);
                self.evolve_insertions(num_insertions, link, branch_id);
            } else {
                let fate = self.sample_tkf_link_fate(branch_length);
                link.fates.push(fate.clone());
                match fate {
                    LinkFate::Deletion => {
                        // link is deleted, do nothing
                    }
                    LinkFate::Homolog(num_children) => {
                        // the surviving homologous link
                        let child_link = TKFLink::new(*child_node, link.length);
                        link.children[branch_id].push(child_link);
                        self.evolve_link_down_tree(link.children[branch_id].last_mut().unwrap());
                        // new insertions
                        self.evolve_insertions(num_children, link, branch_id);
                    }
                    LinkFate::NonHomolog(num_children) => {
                        // original link is deleted, do not evolve_link_down_tree
                        // new insertions
                        self.evolve_insertions(num_children, link, branch_id);
                    }
                }
            }
        }
    }

    fn build_msa_links(&self) -> Vec<TKFLink> {
        let mut root_links = self.build_root_links();
        for link in root_links.iter_mut() {
            self.evolve_link_down_tree(link);
        }
        root_links
    }

    /// Used for the conversion of the link structure to an MSA.
    /// Inserts gaps in the MSA
    ///
    fn insertion_gaps(&self, length: usize, msa: &mut Seqs, insertion_node: &NodeIdx) {
        self.insertion_gaps_subtree(length, msa, &self.tree.root, insertion_node);
    }

    fn insertion_gaps_subtree(
        &self,
        length: usize,
        msa: &mut Seqs,
        subtree_node: &NodeIdx,
        insertion_node: &NodeIdx,
    ) {
        if subtree_node == insertion_node {
            return;
        }
        msa.get_mut(subtree_node)
            .unwrap()
            .extend_from_slice(&vec![NOTHING_CHAR; length]);
        msa.get_mut(subtree_node)
            .unwrap()
            .push(FRAGMENT_BOUNDARY_CHAR);
        for child_node in self.tree.children(subtree_node) {
            self.insertion_gaps_subtree(length, msa, child_node, insertion_node);
        }
    }

    // TODO this shoudl return a simulation result
    pub fn simulate_msa(&self) -> (Seqs, f64, Vec<usize>) {
        *self.cumulative_logl.borrow_mut() = 0.0;
        let links = self.build_msa_links();
        let msa = self.links_to_msa(&links);
        let fragmentation = self.get_fragmentation(&msa);
        (msa, *self.cumulative_logl.borrow(), fragmentation)
    }

    fn get_fragmentation(&self, msa: &Seqs) -> Vec<usize> {
        let root_seq = msa.get(&self.tree.root).unwrap();
        let mut fragmentation = Vec::new();
        for (i, &c) in root_seq.iter().enumerate() {
            if c == FRAGMENT_BOUNDARY_CHAR {
                fragmentation.push(i - fragmentation.len());
            }
        }
        fragmentation
    }

    fn links_to_msa(&self, links: &Vec<TKFLink>) -> Seqs {
        let mut msa: Seqs = HashMap::new();
        for node in self.tree.preorder() {
            msa.insert(*node, Vec::new());
        }
        for link in links {
            self.append_link_to_msa(link, &mut msa);
        }
        msa
    }

    fn msa_to_alignment(&self, msa: &Seqs) -> MASA {
        let records = msa
            .iter()
            .map(|(node, seq)| {
                record!(
                    &self.tree.node(node).id,
                    &seq.iter()
                        .filter(|x| **x != FRAGMENT_BOUNDARY_CHAR)
                        .map(|x| {
                            if *x == NOTHING_CHAR {
                                &DELETION_CHAR
                            } else {
                                x
                            }
                        })
                        .cloned()
                        .collect::<Vec<u8>>()
                )
            })
            .collect();

        let seqs = Sequences::new(records);
        MASA::from_aligned_with_ancestral(seqs, &self.tree).unwrap()
    }

    fn append_link_to_msa(&self, link: &TKFLink, msa: &mut Seqs) {
        let mut insertions = vec![link];
        while !insertions.is_empty() {
            let mut tree_stack = vec![insertions.remove(0)];
            while !tree_stack.is_empty() {
                let current_link = tree_stack.remove(0);
                if current_link.is_immortal {
                    for branch_child in &current_link.children {
                        for (child_id, child_link) in branch_child.iter().enumerate() {
                            if child_id == 0 {
                                tree_stack.push(child_link);
                            } else {
                                insertions.insert(0, child_link);
                            }
                        }
                    }
                } else {
                    msa.get_mut(&current_link.node)
                        .unwrap()
                        .extend_from_slice(&vec![AMB_CHAR; current_link.length]);
                    msa.get_mut(&current_link.node)
                        .unwrap()
                        .push(FRAGMENT_BOUNDARY_CHAR);
                    // normal link
                    if current_link.is_insertion {
                        // this should not be called on the root
                        self.insertion_gaps(current_link.length, msa, &current_link.node);
                    }
                    for (branch_id, child_links) in current_link.children.iter().enumerate() {
                        match &current_link.fates[branch_id] {
                            LinkFate::Homolog(_) => {
                                tree_stack.push(&child_links[0]);
                                for l in child_links.iter().skip(1) {
                                    insertions.insert(0, l); // i think this inserts like [0, 5, 4, 3, 2, 1] perhaps we want [0, 1, 2, 3, 4, 5]
                                }
                            }
                            LinkFate::Deletion => {
                                //insert gaps for all descendants
                                let child_node = self.tree.children(&current_link.node)[branch_id];
                                for descendant_node in self.tree.preorder_subroot(&child_node) {
                                    msa.get_mut(&descendant_node).unwrap().extend_from_slice(
                                        &vec![DELETION_CHAR; current_link.length],
                                    );
                                    msa.get_mut(&descendant_node)
                                        .unwrap()
                                        .push(FRAGMENT_BOUNDARY_CHAR);
                                }
                            }
                            LinkFate::NonHomolog(_) => {
                                for l in child_links {
                                    insertions.insert(0, l); // i think this inserts like [0, 5, 4, 3, 2, 1] perhaps we want [0, 1, 2, 3, 4, 5]
                                }
                                let child_node = self.tree.children(&current_link.node)[branch_id];
                                for descendant_node in self.tree.preorder_subroot(&child_node) {
                                    msa.get_mut(&descendant_node).unwrap().extend_from_slice(
                                        &vec![DELETION_CHAR; current_link.length],
                                    );
                                    msa.get_mut(&descendant_node)
                                        .unwrap()
                                        .push(FRAGMENT_BOUNDARY_CHAR);
                                }
                            }
                        };
                    }
                }
            }
        }
    }
}

fn homolog_prob_integrated(mu: f64, time: f64) -> f64 {
    (-mu * time).exp()
}

fn non_homolog_prob_integrated(mu: f64, beta: f64, time: f64) -> f64 {
    1.0 - (-mu * time).exp() - mu * beta
}

/// In the TKF model n is at least 1 (the original link)
fn homolog_prob(n: usize, lambda: f64, mu: f64, beta: f64, time: f64) -> f64 {
    let h1_val = h1(lambda, mu, beta, time);
    h1_val * (lambda * beta).powi((n - 1) as i32)
}

/// In the TKF model n is at least 1 bc otherwise we should have used n0
fn non_homolog_prob(n: usize, lambda: f64, mu: f64, beta: f64, time: f64) -> f64 {
    let t1 = 1.0 - (-mu * time).exp() - mu * beta;
    let t2 = 1.0 - lambda * beta;
    let t3 = (lambda * beta).powi((n - 1) as i32);
    t1 * t2 * t3
}

/// In the TKF model n is at least 1 because the immortal link cannot die
fn immortal_prob(n: usize, lambda: f64, beta: f64) -> f64 {
    (1.0 - lambda * beta) * (lambda * beta).powi((n - 1) as i32)
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod private_tests {
    use approx::assert_relative_eq;
    use rstest::rstest;

    use crate::phylo_info::PhyloInfo;
    use crate::random::DefaultGenerator;
    use crate::substitution_models::{SubstModel, JC69};
    use crate::tkf_model::TKF92FixedIndelCostBuilder;
    use crate::tkf_model::{beta, n0, TKF91IndelCostBuilder, TKF91IndelModel, TKF92IndelModel};
    use crate::tree::tree_parser::from_newick;

    use super::*;

    #[rstest]
    #[case(0.1, 0.2, 0.5)]
    #[case(0.5, 0.7, 1.0)]
    #[case(1.0, 1.5, 0.0001)]
    fn tkf_integrated_probs(#[case] lambda: f64, #[case] mu: f64, #[case] time: f64) {
        // arrange
        let beta = beta(lambda, mu, time);
        //act
        let homolog_integrated = homolog_prob_integrated(mu, time);
        let non_homolog_integrated = non_homolog_prob_integrated(mu, beta, time);
        let n0 = n0(mu, beta);
        // assert
        assert_relative_eq!(
            homolog_integrated + non_homolog_integrated + n0,
            1.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn tkf_homlog_probs() {
        let lambda = 0.5;
        let mu = 0.7;
        let time = 1.23;
        let beta = beta(lambda, mu, time);
        let n = 4;

        let prob = homolog_prob(n, lambda, mu, beta, time);
        let true_prob =
            (-mu * time).exp() * (1.0 - lambda * beta) * (lambda * beta).powi((n - 1) as i32);
        assert_relative_eq!(prob, true_prob);
    }

    #[test]
    fn tkf_non_homolog_probs() {
        let lambda = 0.5;
        let mu = 0.7;
        let time = 1.23;
        let beta = beta(lambda, mu, time);
        let n = 4;

        let prob = non_homolog_prob(n, lambda, mu, beta, time);
        let true_prob = (1.0 - (-mu * time).exp() - mu * beta)
            * (1.0 - lambda * beta)
            * (lambda * beta).powi((n - 1) as i32);
        assert_relative_eq!(prob, true_prob);
    }

    #[test]
    fn tkf_homolog_probs_sum_to_integrated() {
        let lambda = 0.5;
        let mu = 0.7;
        let time = 1.0;
        let beta = beta(lambda, mu, time);
        let homolog_integrated = homolog_prob_integrated(mu, time);

        let mut homolog_sum = 0.0;
        for n in 1..100 {
            homolog_sum += homolog_prob(n, lambda, mu, beta, time);
        }

        assert_relative_eq!(homolog_sum, homolog_integrated, epsilon = 1e-10);
    }

    #[test]
    fn tkf_non_homolog_probs_sum_to_integrated() {
        let lambda = 0.5;
        let mu = 0.7;
        let time = 1.0;
        let beta = beta(lambda, mu, time);
        let non_homolog_integrated = non_homolog_prob_integrated(mu, beta, time);

        let mut non_homolog_sum = 0.0;
        for n in 1..100 {
            non_homolog_sum += non_homolog_prob(n, lambda, mu, beta, time);
        }

        assert_relative_eq!(non_homolog_sum, non_homolog_integrated, epsilon = 1e-10);
    }

    #[test]
    fn tkf92_simulate() {
        let lambda = 1.1;
        let mu = 1.2;
        let r = 0.6;
        let jc69 = SubstModel::<JC69>::new(&[], &[]);
        let tkf_model = TKF92IndelModel::new(lambda, mu, r);
        let tree =
            from_newick("((A_:0.5,B_:0.5)AB:0.7,(C_:0.6,D_:0.6)CD:0.6)R_;").unwrap()[0].clone();
        let simulator = TKFMSASimulator::new(
            tkf_model,
            jc69,
            tree.clone(),
            DefaultGenerator::default(),
            12, // max insertion length
        );
        let (msa, logl, fragmentation): (Seqs, f64, Vec<usize>) = simulator.simulate_msa();
        assert_eq!(msa.len(), 7);
        for (node, seq) in msa.iter() {
            println!(
                "Node {}: {}",
                tree.node(node).id,
                String::from_utf8_lossy(seq)
            );
        }
        println!(
            "Fragmentation points in the root sequence: {:?}",
            fragmentation
        );
        let alignment = simulator.msa_to_alignment(&msa);
        println!("Alignment:\n{}", alignment);
        let phylo = PhyloInfo {
            msa: alignment,
            tree,
        };
        let cost = TKF92FixedIndelCostBuilder::new(lambda, mu, r, fragmentation, phylo)
            .build()
            .unwrap()
            .logl();
        assert_relative_eq!(logl, cost, epsilon = 1e-10);
    }

    #[test]
    fn tkf91_simulate() {
        let jc69 = SubstModel::<JC69>::new(&[], &[]);
        let tkf_model = TKF91IndelModel::default();
        let lambda = tkf_model.lambda();
        let mu = tkf_model.mu();
        let tree =
            from_newick("((A_:0.5,B_:0.5)AB:0.7,(C_:0.6,D_:0.6)CD:0.6)R_;").unwrap()[0].clone();
        let simulator = TKFMSASimulator::new(
            tkf_model,
            jc69,
            tree.clone(),
            DefaultGenerator::default(),
            12, // max insertion length
        );
        let (msa, logl, fragmentation): (Seqs, f64, Vec<usize>) = simulator.simulate_msa();
        assert_eq!(msa.len(), 7);

        // In TKF91 every residue is its own independent link (fragment length == 1), so the
        // fragmentation must be exactly [1, 2, 3, ..., n_cols].
        let n_cols = fragmentation.len();
        let expected_fragmentation: Vec<usize> = (1..=n_cols).collect();
        assert_eq!(
            fragmentation, expected_fragmentation,
            "TKF91 fragmentation must have every column as its own block"
        );

        // The simulation logl must equal the TKF91 indel cost for the same alignment.
        let alignment = simulator.msa_to_alignment(&msa);
        let phylo = PhyloInfo {
            msa: alignment,
            tree,
        };
        let cost = TKF91IndelCostBuilder::new(lambda, mu, phylo)
            .build()
            .unwrap()
            .logl();
        assert_relative_eq!(logl, cost, epsilon = 1e-10);
    }
}
