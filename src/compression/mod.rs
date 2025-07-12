//! Advanced model compression techniques

pub mod pruning;
pub mod knowledge_distillation;
pub mod low_rank;
pub mod structured;

pub use pruning::*;
pub use knowledge_distillation::*;
pub use low_rank::*;
pub use structured::*; 