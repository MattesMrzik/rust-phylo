use std::cell::RefCell;

use bio::io::fasta::Record;
use hashbrown::HashMap;
use log::warn;
use rand::{Rng, RngCore, SeedableRng};
use rand_distr::{Distribution, Geometric};

use crate::alignment::{AlignmentSimulation, AncestralAlignment, Sequences};
use crate::alphabets::{AMB_CHAR, GAP};
use crate::random::RandomGenerator;
use crate::record_wo_desc as record;
use crate::substitution_models::{QMatrix, SubstModel, SubstitutionSimulatorBuilder};
use crate::tkf_model::{beta, h1, n0, TKFModel};
use crate::tree::{NodeIdx, NodeIdx::Internal, NodeIdx::Leaf, Tree};

impl<T, R> AlignmentSimulation for TKFIndelMSASimulator<T, R>
where
    T: TKFModel + FragmentSampler,
    R: Rng + SeedableRng + RngCore,
{
    fn simulate_ancestral_alignment<AA: AncestralAlignment>(&self) -> AA {
        let result = self.simulate_msa();
        let TKFMSASimulationResult {
            msa,
            fragmentation: _,
            logl: _,
        } = result;
        msa
    }
}

/// Simulates a full TKF process: first indels (TKF) then substitutions.
///
/// The simulator runs the [indel simulator](TKFIndelMSASimulator) to obtain an MSA with gaps, then
/// simulates substitutions along the same tree for the number of columns
/// produced by the indel simulation and finally uses the indel MSA as a mask
/// to replace characters with gaps where the indel process produced deletions.
pub struct TKFMSASimulator<Q, T, R>
where
    Q: QMatrix,
    T: TKFModel + FragmentSampler,
    R: Rng + SeedableRng + RngCore,
{
    indel_sim: TKFIndelMSASimulator<T, R>,
    subst_model: SubstModel<Q>,
}

impl<Q, T, R> TKFMSASimulator<Q, T, R>
where
    Q: QMatrix,
    T: TKFModel + FragmentSampler,
    R: Rng + SeedableRng + RngCore + Clone,
{
    /// Create a new TKFMSASimulator with the given indel model, substitution model, tree, RNG and
    /// max insertion length (i.e., the max number of inserted links (=fragments) in a single event).
    pub fn new(
        indel_model: T,
        subst_model: SubstModel<Q>,
        tree: Tree,
        rng: RandomGenerator<R>,
        max_insertion_length: usize,
    ) -> Self {
        let indel_sim = TKFIndelMSASimulator::new(indel_model, tree, rng, max_insertion_length);
        Self {
            indel_sim,
            subst_model,
        }
    }
}

impl<Q, T, R> AlignmentSimulation for TKFMSASimulator<Q, T, R>
where
    Q: QMatrix,
    T: TKFModel + FragmentSampler,
    R: Rng + SeedableRng + RngCore + Clone,
{
    /// Simulate an ancestral alignment with indels and substitutions.
    ///
    /// The returned `TKFMSASimulationResult` contains the final alignment where
    /// positions deleted by the indel process are gaps and surviving positions
    /// have characters sampled by the substitution model.
    fn simulate_ancestral_alignment<AA: AncestralAlignment>(&self) -> AA {
        // First, simulate indels
        let indel_result = self.indel_sim.simulate_msa::<AA>();
        let indel_msa = indel_result.msa();

        // Second, substitution simulation
        let aln_len = indel_msa.len();

        let rng_clone = self.indel_sim.rng.borrow().clone();
        let subst_builder = SubstitutionSimulatorBuilder::new(
            self.subst_model.clone(),
            self.indel_sim.tree.clone(),
            rng_clone,
        )
        .alignment_length(aln_len)
        .build()
        .unwrap(); // will not fail since we set the alignment_length
        let subst_msa: AA = subst_builder.simulate_ancestral_alignment();
        // Third, mask the substitution msa with gaps from the indel msa
        // Construct a combined sequences vector (including ancestral records)
        let mut combined_records: Vec<Record> = Vec::new();
        for node in self.indel_sim.tree.preorder() {
            let id = self.indel_sim.tree.node(node).id.clone();
            // get mask seq (from indel msa) and subst seq (from substitution msa)
            let mask_mapping = match node {
                Internal(_) => indel_msa.ancestral_map(node),
                Leaf(_) => indel_msa.leaf_map(node),
            };
            let subst_seq = match node {
                Leaf(_) => subst_msa.seqs().record_by_id(&id).seq(),
                Internal(_) => subst_msa.ancestral_seqs().record_by_id(&id).seq(),
            };

            debug_assert!(
                mask_mapping.len() == subst_seq.len(),
                "Mask and substitution sequences must be the same length"
            );
            // apply mask: if mask is a gap, put a gap; otherwise keep the subst character
            let final_seq: Vec<u8> = mask_mapping
                .iter()
                .zip(subst_seq.iter())
                .map(
                    |(mask, subst_char)| {
                        if mask.is_some() {
                            *subst_char
                        } else {
                            GAP
                        }
                    },
                )
                .collect();

            combined_records.push(record!(&id, &final_seq));
        }

        let seqs = Sequences::new(combined_records);
        AA::from_aligned_with_ancestral(seqs, &self.indel_sim.tree).unwrap()
    }
}

/// Abstracts over how a single fragment's length is sampled.
///
/// [TKF92](`crate::tkf_model::TKF92IndelModel`) draws from a geometric distribution parameterised by `r`.
/// [TKF91](`crate::tkf_model::TKF91IndelModel`) always returns length 1 (each residue is its own independent link).
///
/// Returns `(length, log_probability)` so that the caller can accumulate
/// the simulation log-likelihood without any model-specific logic.
pub trait FragmentSampler {
    fn sample_fragment_length<R: Rng>(&self, rng: &mut R) -> (usize, f64);
}

/// Since these sequences are built incrementally, we can't use the [`Sequences`] which hold immutable [records](`record`).
type Seqs = HashMap<NodeIdx, Vec<u8>>;

/// Simulates the indel process under a [TKFModel](crate::tkf_model::TKFModel) to produce an MSA
/// containing only [`GAP`]s and [`AMB_CHAR`]s, representing the indel history.
/// The `max_insertion_length` parameter controls the maximum number of inserted links (=fragments)
/// that can be produced in a single insertion event.
pub struct TKFIndelMSASimulator<T: TKFModel + FragmentSampler, R: Rng + SeedableRng + RngCore> {
    indel_model: T,
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

pub struct TKFMSASimulationResult<AA: AncestralAlignment> {
    msa: AA,
    fragmentation: Vec<usize>,
    logl: f64,
}

impl<AA: AncestralAlignment> TKFMSASimulationResult<AA> {
    pub fn msa(&self) -> &AA {
        &self.msa
    }

    pub fn fragmentation(&self) -> &Vec<usize> {
        &self.fragmentation
    }

    pub fn logl(&self) -> f64 {
        self.logl
    }
}

impl<T, R> TKFIndelMSASimulator<T, R>
where
    T: TKFModel + FragmentSampler,
    R: Rng + SeedableRng + RngCore,
{
    pub fn new(
        indel_model: T,
        tree: Tree,
        rng: RandomGenerator<R>,
        max_insertion_length: usize,
    ) -> Self {
        Self {
            indel_model,
            tree,
            cumulative_logl: RefCell::new(0.0),
            rng: RefCell::new(rng),
            max_insertion_length,
        }
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
        // didn't return in the loop, capping insertion at length max_insertion_length
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
        // didn't return in the loop, capping insertion at length max_insertion_length
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
        let mut cumulative_prob = 0.0;
        for n in 1..self.max_insertion_length {
            let immortal_prob = immortal_prob(n, lambda, beta);
            cumulative_prob += immortal_prob;
            if uniform_sample < cumulative_prob {
                *self.cumulative_logl.borrow_mut() += immortal_prob.ln();
                return n - 1;
            }
        }
        let immortal_prob = 1.0 - cumulative_prob;
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
        let seq = msa.get_mut(subtree_node).unwrap();
        let new_len = seq.len() + length;
        seq.resize(new_len, GAP);
        for child_node in self.tree.children(subtree_node) {
            self.insertion_gaps_subtree(length, msa, child_node, insertion_node);
        }
    }

    pub fn simulate_msa<AA: AncestralAlignment>(&self) -> TKFMSASimulationResult<AA> {
        *self.cumulative_logl.borrow_mut() = 0.0;
        let links = self.build_msa_links();
        let (raw_msa, fragmentation) = self.links_to_msa(&links);
        let msa: AA = self.msa_to_alignment(&raw_msa);
        TKFMSASimulationResult {
            msa,
            fragmentation,
            logl: *self.cumulative_logl.borrow(),
        }
    }

    fn links_to_msa(&self, links: &Vec<TKFLink>) -> (Seqs, Vec<usize>) {
        let mut msa: Seqs = HashMap::new();
        for node in self.tree.preorder() {
            msa.insert(*node, Vec::new());
        }
        let mut fragmentation: Vec<usize> = Vec::new();
        for link in links {
            self.append_link_to_msa(link, &mut msa, &mut fragmentation);
        }
        (msa, fragmentation)
    }

    fn msa_to_alignment<AA: AncestralAlignment>(&self, msa: &Seqs) -> AA {
        let records = msa
            .iter()
            .map(|(node, seq)| record!(&self.tree.node(node).id, seq))
            .collect();

        let seqs = Sequences::new(records);
        AA::from_aligned_with_ancestral(seqs, &self.tree).unwrap()
    }

    fn append_link_to_msa(&self, link: &TKFLink, msa: &mut Seqs, fragmentation: &mut Vec<usize>) {
        let mut insertions = vec![link];
        while !insertions.is_empty() {
            let mut tree_stack = vec![insertions.remove(0)];
            while !tree_stack.is_empty() {
                let current_link = tree_stack.remove(0);
                if current_link.is_immortal {
                    self.dispatch_immortal_children(current_link, &mut tree_stack, &mut insertions);
                } else {
                    self.process_link(
                        current_link,
                        msa,
                        fragmentation,
                        &mut tree_stack,
                        &mut insertions,
                    );
                }
            }
        }
    }

    /// Pushes the children of an immortal link onto the traversal stacks.
    ///
    /// The first child of each branch continues the homolog tree traversal (`tree_stack`) which is
    /// the immortal surviving; any additional children are new insertions and go onto the `insertions` queue.
    fn dispatch_immortal_children<'a>(
        &self,
        link: &'a TKFLink,
        tree_stack: &mut Vec<&'a TKFLink>,
        insertions: &mut Vec<&'a TKFLink>,
    ) {
        for branch_child in &link.children {
            for (child_id, child_link) in branch_child.iter().enumerate() {
                if child_id == 0 {
                    tree_stack.push(child_link);
                } else {
                    insertions.insert(0, child_link);
                }
            }
        }
    }

    /// Writes one non-immortal link's contribution to the MSA and updates the traversal stacks.
    ///
    /// - Appends `AMB_CHAR * length` to the owning node.
    /// - Records a fragmentation boundary for root-owned links and insertion links.
    /// - Fills `NOTHING_CHAR` gaps into all non-owning nodes for insertion links.
    /// - Applies each branch fate: queues homolog/insertion children, or fills deletion gaps.
    fn process_link<'a>(
        &self,
        link: &'a TKFLink,
        msa: &mut Seqs,
        fragmentation: &mut Vec<usize>,
        tree_stack: &mut Vec<&'a TKFLink>,
        insertions: &mut Vec<&'a TKFLink>,
    ) {
        let seq = msa.get_mut(&link.node).unwrap();
        let new_len = seq.len() + link.length;
        seq.resize(new_len, AMB_CHAR);
        if link.node == self.tree.root || link.is_insertion {
            let fragment_boundary = link.length + fragmentation.last().unwrap_or(&0);
            fragmentation.push(fragment_boundary);
        }
        if link.is_insertion {
            self.insertion_gaps(link.length, msa, &link.node);
        }
        for (branch_id, child_links) in link.children.iter().enumerate() {
            self.apply_fate(link, branch_id, child_links, msa, tree_stack, insertions);
        }
    }

    /// Applies a single branch fate for a non-immortal link.
    ///
    /// - `Homolog`: queues the surviving child onto `tree_stack` and its insertions onto `insertions`.
    /// - `Deletion` / `NonHomolog`: fills deletion gaps for all descendants; `NonHomolog` also
    ///   queues the new insertion children.
    fn apply_fate<'a>(
        &self,
        link: &'a TKFLink,
        branch_id: usize,
        child_links: &'a [TKFLink],
        msa: &mut Seqs,
        tree_stack: &mut Vec<&'a TKFLink>,
        insertions: &mut Vec<&'a TKFLink>,
    ) {
        match &link.fates[branch_id] {
            LinkFate::Homolog(_) => {
                tree_stack.push(&child_links[0]);
                for l in child_links.iter().skip(1) {
                    insertions.insert(0, l);
                }
            }
            LinkFate::Deletion => {
                let child_node = self.tree.children(&link.node)[branch_id];
                for descendant_node in self.tree.preorder_subroot(&child_node) {
                    let seq = msa.get_mut(&descendant_node).unwrap();
                    let new_len = seq.len() + link.length;
                    seq.resize(new_len, GAP);
                }
            }
            LinkFate::NonHomolog(_) => {
                for l in child_links {
                    insertions.insert(0, l);
                }
                let child_node = self.tree.children(&link.node)[branch_id];
                for descendant_node in self.tree.preorder_subroot(&child_node) {
                    let seq = msa.get_mut(&descendant_node).unwrap();
                    let new_len = seq.len() + link.length;
                    seq.resize(new_len, GAP);
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

/// In the TKF model n is at least 1 because otherwise we should have used n0
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
    use hashbrown::HashSet;
    use rstest::rstest;

    use crate::alignment::{Alignment, AncestralAlignment, MASA};
    use crate::phylo_info::PhyloInfo;
    use crate::random::DefaultGenerator;
    use crate::substitution_models::{dna_models::GTR, SubstModel};
    use crate::tkf_model::{
        beta, n0, TKF91IndelCostBuilder, TKF91IndelModel, TKF92FixedIndelCostBuilder,
        TKF92IndelModel,
    };
    use crate::tree;

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
    fn tkf_homolog_probs() {
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
    fn tkf92_indel_simulate() {
        let lambda = 1.1;
        let mu = 1.2;
        let r = 0.6;
        let tkf_model = TKF92IndelModel::new(lambda, mu, r);
        let tree = tree!("((A:0.5,B:0.5)AB:0.7,(C:0.6,D:0.6)CD:0.6)R;");

        // Should be so large that it doesn't kick in. Because otherwise the
        // accumulated logl would not match the clean cost. Since event probabilities
        // are summed together for the case of drawing a max_insertion_length.
        let max_insertion_length = 100;
        let simulator = TKFIndelMSASimulator::new(
            tkf_model,
            tree.clone(),
            DefaultGenerator::new(41),
            max_insertion_length,
        );
        let result: TKFMSASimulationResult<MASA> = simulator.simulate_msa();
        let alignment = result.msa();
        assert_eq!(alignment.seq_count() + alignment.ancestral_seqs().len(), 7);
        let phylo = PhyloInfo {
            msa: alignment.clone(),
            tree,
        };
        assert!(
            phylo.check_dollos_constraint().is_ok(),
            "Simulated alignment must satisfy Dollo's constraint (no re-gain of characters)"
        );
        let cost =
            TKF92FixedIndelCostBuilder::new(lambda, mu, r, result.fragmentation().clone(), phylo)
                .build()
                .unwrap()
                .logl();
        assert_relative_eq!(result.logl(), cost, epsilon = 1e-10);
    }

    #[test]
    fn tkf91_indel_simulate() {
        let tkf_model = TKF91IndelModel::default();
        let lambda = tkf_model.lambda();
        let mu = tkf_model.mu();
        let tree = tree!("((A:0.5,B:0.5)AB:0.7,(C:0.6,D:0.6)CD:0.6)R;");

        // Should be so large that it doesn't kick in. Because otherwise the
        // accumulated logl would not match the clean cost. Since event probabilities
        // are summed together for the case of drawing a max_insertion_length.
        let max_insertion_length = 100;
        let simulator = TKFIndelMSASimulator::new(
            tkf_model,
            tree.clone(),
            DefaultGenerator::new(41),
            max_insertion_length,
        );
        let result: TKFMSASimulationResult<MASA> = simulator.simulate_msa();
        let alignment = result.msa();
        assert_eq!(alignment.seq_count() + alignment.ancestral_seqs().len(), 7);

        // In TKF91 every residue is its own independent link (fragment length == 1), so the
        // fragmentation must be exactly [1, 2, 3, ..., n_cols].
        let n_cols = result.fragmentation().len();
        let expected_fragmentation: Vec<usize> = (1..=n_cols).collect();
        assert_eq!(
            result.fragmentation(),
            &expected_fragmentation,
            "TKF91 fragmentation must have every column as its own block"
        );

        // The simulation logl must equal the TKF91 indel cost for the same alignment.
        let phylo = PhyloInfo {
            msa: alignment.clone(),
            tree,
        };
        assert!(
            phylo.check_dollos_constraint().is_ok(),
            "Simulated alignment must satisfy Dollo's constraint (no re-gain of characters)"
        );
        let cost = TKF91IndelCostBuilder::new(lambda, mu, phylo)
            .build()
            .unwrap()
            .logl();
        assert_relative_eq!(result.logl(), cost, epsilon = 1e-10);
    }

    /// In the [`tkf92_simulation`] test, only A <-> T and G <-> C transitions are allowed.
    /// Checks that the simulation respects the mutation constraints of the GTR model used.
    /// So each column must contain at most two unique characters, which must be a valid pair.
    #[cfg(test)]
    fn check_mutation_constraints(msa: &MASA, tree: &Tree) {
        for col_idx in 0..msa.len() {
            let mut col_chars = HashSet::new();
            for node_idx in tree.preorder() {
                let node_id = tree.node_id(node_idx);
                let seq = match node_idx {
                    Leaf(_) => msa.seqs().record_by_id(node_id).seq(),
                    Internal(_) => msa.ancestral_seqs().record_by_id(node_id).seq(),
                };
                let map = match node_idx {
                    Internal(_) => msa.ancestral_map(node_idx),
                    Leaf(_) => msa.leaf_map(node_idx),
                };

                if let Some(pos) = map[col_idx] {
                    col_chars.insert(seq[pos]);
                }
            }
            assert!(
                col_chars.len() <= 2,
                "Column {} has too many unique characters: {:?}",
                col_idx,
                col_chars
            );
            // Verify specific pairings if multiple characters exist
            if col_chars.len() == 2 {
                let chars: Vec<u8> = col_chars.into_iter().collect();
                let c1 = chars[0];
                let c2 = chars[1];
                let valid_pair = matches!(
                    (c1, c2),
                    (b'A', b'T') | (b'T', b'A') | (b'C', b'G') | (b'G', b'C')
                );
                assert!(
                    valid_pair,
                    "Invalid mutation pair in column {}: {} and {}",
                    col_idx, c1 as char, c2 as char
                );
            }
        }
    }

    #[test]
    fn tkf92_simulation() {
        let tree = tree!(
            "((((A:1.0,B:1.0)I1:0.5,C:1.5)I2:0.5,D:2.0)I3:0.5,((E:1.0,F:1.0)I4:0.5,G:1.5)I5:0.5)R;"
        );

        let freqs = [0.25, 0.25, 0.25, 0.25];
        let params = [0.0, 1.0, 0.0, 0.0, 1.0, 0.0];
        let subst_model = SubstModel::<GTR>::new(&freqs, &params);

        let lambda = 0.19;
        let mu = 0.2;
        let r = 0.8;
        let tkf_model = TKF92IndelModel::new(lambda, mu, r);

        let max_insertion_length = 50;
        let simulator = TKFMSASimulator::new(
            tkf_model,
            subst_model,
            tree.clone(),
            DefaultGenerator::new(123),
            max_insertion_length,
        );

        let msa = simulator.simulate_ancestral_alignment::<MASA>();

        assert_eq!(msa.seq_count() + msa.ancestral_seqs().len(), 13);
        assert_eq!(msa.seq_count(), 7); // A, B, C, D, E, F, G
        assert!(msa.len() > 1);

        check_mutation_constraints(&msa, &tree);

        let phylo = PhyloInfo {
            msa: msa.clone(),
            tree,
        };
        assert!(
            phylo.check_dollos_constraint().is_ok(),
            "Simulated alignment must satisfy Dollo's constraint (no re-gain of characters)"
        );
    }

    #[test]
    fn tkf92_indel_trait_consistency() {
        let lambda = 0.2;
        let mu = 0.3;
        let r = 0.99;
        let tkf_model = TKF92IndelModel::new(lambda, mu, r);
        let tree = tree!("(A:1.0,B:1.0)R:1.0;");
        let max_len = 2;
        let seed = 123;
        let simulator1 = TKFIndelMSASimulator::new(
            tkf_model.clone(),
            tree.clone(),
            DefaultGenerator::new(seed),
            max_len,
        );
        let simulator2 =
            TKFIndelMSASimulator::new(tkf_model, tree, DefaultGenerator::new(seed), max_len);
        let result1 = simulator1.simulate_msa::<MASA>();
        let msa2: MASA = simulator2.simulate_ancestral_alignment();
        assert_eq!(result1.msa().to_string(), msa2.to_string());
    }

    #[test]
    fn tkf92_indel_capping() {
        let lambda = 0.19;
        let mu = 0.2;
        let r = 0.5;
        let tkf_model = TKF92IndelModel::new(lambda, mu, r);
        let tree = tree!("(A:100.0,B:100.0)R:1.0;");
        let max_len = 0; // Cap at 0 insertions on branches
        let simulator =
            TKFIndelMSASimulator::new(tkf_model, tree.clone(), DefaultGenerator::new(123), max_len);
        let result = simulator.simulate_msa::<MASA>();
        let msa = result.msa();
        // With max_len = 0, no insertions can happen on branches.
        // Thus, every column in the MSA must be have a char at the root.
        let root_map = msa.ancestral_map(&tree.root);
        for (col_idx, site) in root_map.iter().enumerate() {
            assert!(
                site.is_some(),
                "Column {} should have a character at the root",
                col_idx
            );
        }
    }
}
