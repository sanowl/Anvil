//! Learning rate schedulers

use crate::error::AnvilResult;

/// Learning rate scheduler trait
pub trait LRScheduler: Send + Sync {
    fn step(&mut self, metric: f32);
    fn get_lr(&self) -> f32;
    fn reset(&mut self);
}

/// Scheduler configuration
#[derive(Debug, Clone)]
pub enum SchedulerConfig {
    StepLR { step_size: usize, gamma: f32 },
    ExponentialLR { gamma: f32 },
    CosineAnnealingLR { t_max: usize, eta_min: f32 },
    ReduceLROnPlateau { factor: f32, patience: usize, threshold: f32 },
}

impl SchedulerConfig {
    pub fn create_scheduler(&self) -> Box<dyn LRScheduler> {
        match self {
            SchedulerConfig::StepLR { step_size, gamma } => {
                Box::new(StepLRScheduler::new(*step_size, *gamma))
            },
            SchedulerConfig::ExponentialLR { gamma } => {
                Box::new(ExponentialLRScheduler::new(*gamma))
            },
            SchedulerConfig::CosineAnnealingLR { t_max, eta_min } => {
                Box::new(CosineAnnealingLRScheduler::new(*t_max, *eta_min))
            },
            SchedulerConfig::ReduceLROnPlateau { factor, patience, threshold } => {
                Box::new(ReduceLROnPlateauScheduler::new(*factor, *patience, *threshold))
            },
        }
    }
}

/// Scheduler types
#[derive(Debug, Clone, Copy)]
pub enum SchedulerType {
    StepLR,
    ExponentialLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    OneCycleLR,
}

/// Step LR Scheduler
pub struct StepLRScheduler {
    step_size: usize,
    gamma: f32,
    current_step: usize,
    current_lr: f32,
    base_lr: f32,
}

impl StepLRScheduler {
    pub fn new(step_size: usize, gamma: f32) -> Self {
        Self {
            step_size,
            gamma,
            current_step: 0,
            current_lr: 1e-3,
            base_lr: 1e-3,
        }
    }
}

impl LRScheduler for StepLRScheduler {
    fn step(&mut self, _metric: f32) {
        self.current_step += 1;
        if self.current_step % self.step_size == 0 {
            self.current_lr *= self.gamma;
        }
    }
    
    fn get_lr(&self) -> f32 {
        self.current_lr
    }
    
    fn reset(&mut self) {
        self.current_step = 0;
        self.current_lr = self.base_lr;
    }
}

/// Exponential LR Scheduler
pub struct ExponentialLRScheduler {
    gamma: f32,
    current_lr: f32,
    base_lr: f32,
}

impl ExponentialLRScheduler {
    pub fn new(gamma: f32) -> Self {
        Self {
            gamma,
            current_lr: 1e-3,
            base_lr: 1e-3,
        }
    }
}

impl LRScheduler for ExponentialLRScheduler {
    fn step(&mut self, _metric: f32) {
        self.current_lr *= self.gamma;
    }
    
    fn get_lr(&self) -> f32 {
        self.current_lr
    }
    
    fn reset(&mut self) {
        self.current_lr = self.base_lr;
    }
}

/// Cosine Annealing LR Scheduler
pub struct CosineAnnealingLRScheduler {
    t_max: usize,
    eta_min: f32,
    current_step: usize,
    current_lr: f32,
    base_lr: f32,
}

impl CosineAnnealingLRScheduler {
    pub fn new(t_max: usize, eta_min: f32) -> Self {
        Self {
            t_max,
            eta_min,
            current_step: 0,
            current_lr: 1e-3,
            base_lr: 1e-3,
        }
    }
}

impl LRScheduler for CosineAnnealingLRScheduler {
    fn step(&mut self, _metric: f32) {
        self.current_step += 1;
        let progress = self.current_step as f32 / self.t_max as f32;
        self.current_lr = self.eta_min + (self.base_lr - self.eta_min) * 
            (1.0 + (std::f32::consts::PI * progress).cos()) / 2.0;
    }
    
    fn get_lr(&self) -> f32 {
        self.current_lr
    }
    
    fn reset(&mut self) {
        self.current_step = 0;
        self.current_lr = self.base_lr;
    }
}

/// Reduce LR on Plateau Scheduler
pub struct ReduceLROnPlateauScheduler {
    factor: f32,
    patience: usize,
    threshold: f32,
    current_lr: f32,
    base_lr: f32,
    best_metric: f32,
    wait_count: usize,
}

impl ReduceLROnPlateauScheduler {
    pub fn new(factor: f32, patience: usize, threshold: f32) -> Self {
        Self {
            factor,
            patience,
            threshold,
            current_lr: 1e-3,
            base_lr: 1e-3,
            best_metric: f32::INFINITY,
            wait_count: 0,
        }
    }
}

impl LRScheduler for ReduceLROnPlateauScheduler {
    fn step(&mut self, metric: f32) {
        if metric < self.best_metric - self.threshold {
            self.best_metric = metric;
            self.wait_count = 0;
        } else {
            self.wait_count += 1;
            if self.wait_count >= self.patience {
                self.current_lr *= self.factor;
                self.wait_count = 0;
            }
        }
    }
    
    fn get_lr(&self) -> f32 {
        self.current_lr
    }
    
    fn reset(&mut self) {
        self.current_lr = self.base_lr;
        self.best_metric = f32::INFINITY;
        self.wait_count = 0;
    }
}