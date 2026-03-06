use std::cell::RefCell;

use hashbrown::HashMap;
use rand::{distr::weighted::WeightedIndex, Rng, RngCore, SeedableRng};

use crate::alignment::{AlignmentSimulation, AncestralAlignment, Sequences};
use crate::random::RandomGenerator;
use crate::record_wo_desc as record;
use crate::substitution_models::{QMatrix, SubstMatrix, SubstModel};
use crate::tree::{NodeIdx, Tree};
use crate::Result;

pub struct SubstitutionSimulatorBuilder<Q: QMatrix, R: Rng + SeedableRng + RngCore> {
    model: SubstModel<Q>,
    tree: Tree,
    rng: RandomGenerator<R>,
    alignment_length: Option<usize>,
}

impl<Q: QMatrix, R: Rng + SeedableRng + RngCore> SubstitutionSimulatorBuilder<Q, R> {
    pub fn new(
        model: crate::substitution_models::SubstModel<Q>,
        tree: Tree,
        rng: RandomGenerator<R>,
    ) -> Self {
        Self {
            model,
            tree,
            rng,
            alignment_length: None,
        }
    }

    pub fn alignment_length(mut self, length: usize) -> Self {
        self.alignment_length = Some(length);
        self
    }

    pub fn build(self) -> Result<SubstitutionSimulator<Q, R>> {
        let alignment_length = self
            .alignment_length
            .ok_or_else(|| crate::Error::Alignment("alignment_length must be set".to_string()))?;

        // Precompute transition matrices P = exp(Q * branch_length) for each non-root node.
        let p_matrices: HashMap<NodeIdx, SubstMatrix> = self
            .tree
            .postorder()
            .iter()
            .filter_map(|idx| self.tree.parent(idx).map(|_| *idx))
            .map(|idx| {
                let blen = self.tree.node(&idx).blen;
                // Reimplement p(time) without requiring the EvoModel trait here.
                // Because otherwise passing the PIP model would be fine, but it isn't
                let qmat = self.model.qmatrix.q().clone();
                let p = if blen > crate::MAX_BLEN {
                    (qmat * crate::MAX_BLEN).exp()
                } else {
                    (qmat * blen).exp()
                };
                (idx, p)
            })
            .collect();

        Ok(SubstitutionSimulator {
            model: self.model,
            tree: self.tree,
            p_matrices,
            rng: RefCell::new(self.rng),
            alignment_length,
        })
    }
}

pub struct SubstitutionSimulator<Q: QMatrix, R: Rng + SeedableRng + RngCore> {
    model: crate::substitution_models::SubstModel<Q>,
    tree: Tree,
    p_matrices: HashMap<NodeIdx, SubstMatrix>,
    rng: RefCell<RandomGenerator<R>>,
    alignment_length: usize,
}

impl<Q: QMatrix, R: Rng + SeedableRng + RngCore> SubstitutionSimulator<Q, R> {
    fn sample_from_freqs(&self) -> usize {
        let freqs = self.model.qmatrix.freqs();
        let dist = WeightedIndex::new(freqs.as_slice()).unwrap();
        self.rng.borrow_mut().sample(&dist)
    }

    fn sample_child(&self, parent_state: usize, p_matrix: &SubstMatrix) -> usize {
        let column = p_matrix.column(parent_state);
        let probs: Vec<f64> = column.iter().cloned().collect();
        let dist = WeightedIndex::new(&probs).unwrap();
        self.rng.borrow_mut().sample(&dist)
    }

    fn state_to_char(&self, state: usize) -> u8 {
        Q::alphabet().symbols()[state]
    }
}

impl<Q: QMatrix, R: Rng + SeedableRng + RngCore> AlignmentSimulation
    for SubstitutionSimulator<Q, R>
{
    fn simulate_ancestral_alignment<AA: AncestralAlignment>(&self) -> AA {
        let mut sequences: HashMap<NodeIdx, Vec<usize>> = HashMap::with_capacity(self.tree.len());

        let root_seq: Vec<usize> = (0..self.alignment_length)
            .map(|_| self.sample_from_freqs())
            .collect();
        sequences.insert(self.tree.root, root_seq);

        for node_idx in self.tree.preorder().iter().skip(1) {
            let parent_idx = self.tree.parent(node_idx).unwrap();
            let parent_seq = sequences.get(&parent_idx).unwrap();
            let p_matrix = self.p_matrices.get(node_idx).unwrap();

            let child_seq: Vec<usize> = parent_seq
                .iter()
                .map(|&parent_state| self.sample_child(parent_state, p_matrix))
                .collect();

            sequences.insert(*node_idx, child_seq);
        }

        let records: Vec<_> = sequences
            .iter()
            .map(|(node_idx, seq)| {
                let id = self.tree.node_id(node_idx);
                let char_seq: Vec<u8> = seq.iter().map(|&s| self.state_to_char(s)).collect();
                record!(id, &char_seq)
            })
            .collect();

        let seqs = Sequences::new(records);
        AA::from_aligned_with_ancestral(seqs, &self.tree).unwrap()
    }
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod tests {
    use super::*;
    use crate::alignment::{Alignment, MASA};
    use crate::random::DefaultGenerator;
    use crate::substitution_models::dna_models::JC69;
    use crate::substitution_models::SubstModel;
    use crate::tree::tree_parser::from_newick;

    #[test]
    fn test_substitution_simulator_basic() {
        let model = SubstModel::<JC69>::new(&[], &[]);
        let tree = from_newick("((A:0.5,B:0.5)AB:0.7,(C:0.6,D:0.6)CD:0.6)R;").unwrap()[0].clone();
        let rng = DefaultGenerator::new(42);

        let simulator = SubstitutionSimulatorBuilder::new(model, tree.clone(), rng)
            .alignment_length(100)
            .build()
            .unwrap();

        let alignment: MASA = simulator.simulate_ancestral_alignment();

        assert_eq!(alignment.len(), 100);
        assert_eq!(alignment.seq_count(), 4);
        assert_eq!(alignment.ancestral_seqs().len(), 3);
    }

    #[test]
    fn test_substitution_simulator_longer_tree() {
        let model = SubstModel::<JC69>::new(&[], &[]);
        let tree = from_newick("((A:2.0,B:2.0)AB:2.0,(C:2.0,D:2.0)CD:2.0)R;").unwrap()[0].clone();
        let rng = DefaultGenerator::new(123);

        let simulator = SubstitutionSimulatorBuilder::new(model, tree.clone(), rng)
            .alignment_length(50)
            .build()
            .unwrap();

        let alignment: MASA = simulator.simulate_ancestral_alignment();

        assert_eq!(alignment.len(), 50);
        assert_eq!(
            alignment.seq_count() + alignment.ancestral_seqs().len(),
            tree.len()
        );
    }

    #[test]
    fn test_substitution_simulator_three_taxa() {
        let model = SubstModel::<JC69>::new(&[], &[]);
        let tree = from_newick("((A:0.1,B:0.1)AB:0.1,C:0.2)R;").unwrap()[0].clone();
        let rng = DefaultGenerator::new(999);

        let simulator = SubstitutionSimulatorBuilder::new(model, tree.clone(), rng)
            .alignment_length(200)
            .build()
            .unwrap();

        let alignment: MASA = simulator.simulate_ancestral_alignment();

        assert_eq!(alignment.len(), 200);
        assert_eq!(alignment.seq_count(), 3);
    }

    #[test]
    fn test_reproducibility() {
        let model = SubstModel::<JC69>::new(&[], &[]);
        let tree = from_newick("((A:0.5,B:0.5)AB:0.7,(C:0.6,D:0.6)CD:0.6)R;").unwrap()[0].clone();

        let rng1 = DefaultGenerator::new(42);
        let rng2 = DefaultGenerator::new(42);

        let simulator1 = SubstitutionSimulatorBuilder::new(model.clone(), tree.clone(), rng1)
            .alignment_length(100)
            .build()
            .unwrap();

        let simulator2 = SubstitutionSimulatorBuilder::new(model, tree.clone(), rng2)
            .alignment_length(100)
            .build()
            .unwrap();

        let alignment1: MASA = simulator1.simulate_ancestral_alignment();
        let alignment2: MASA = simulator2.simulate_ancestral_alignment();

        assert_eq!(
            alignment1.to_string(),
            alignment2.to_string(),
            "Same seed should produce identical alignments"
        );
    }

    #[test]
    fn test_builder_requires_alignment_length() {
        let model = SubstModel::<JC69>::new(&[], &[]);
        let tree = from_newick("((A:0.5,B:0.5)AB:0.7);").unwrap()[0].clone();
        let rng = DefaultGenerator::new(42);

        let result = SubstitutionSimulatorBuilder::new(model, tree, rng).build();

        assert!(result.is_err());
    }
}
