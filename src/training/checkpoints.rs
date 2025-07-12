//! Checkpoint management for training

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use crate::{
    error::{AnvilError, AnvilResult},
    training::metrics::TrainingMetrics,
};

/// Training checkpoint
#[derive(Debug, Clone)]
pub struct Checkpoint {
    pub epoch: usize,
    pub step: usize,
    pub loss: f32,
    pub learning_rate: f32,
    pub optimizer_state: HashMap<String, Vec<u8>>,
    pub model_state: HashMap<String, Vec<u8>>,
    pub metrics: TrainingMetrics,
}

/// Checkpoint manager
pub struct CheckpointManager {
    checkpoint_dir: PathBuf,
    max_checkpoints: usize,
}

impl CheckpointManager {
    pub fn new<P: AsRef<Path>>(checkpoint_dir: P) -> Self {
        Self {
            checkpoint_dir: checkpoint_dir.as_ref().to_path_buf(),
            max_checkpoints: 5,
        }
    }
    
    pub fn with_max_checkpoints(mut self, max_checkpoints: usize) -> Self {
        self.max_checkpoints = max_checkpoints;
        self
    }
    
    pub async fn save_checkpoint(&self, checkpoint: &Checkpoint, name: &str) -> AnvilResult<()> {
        // Create checkpoint directory if it doesn't exist
        std::fs::create_dir_all(&self.checkpoint_dir)
            .map_err(|e| AnvilError::ComputationError(format!("Failed to create checkpoint directory: {}", e)))?;
        
        let checkpoint_path = self.checkpoint_dir.join(format!("{}.ckpt", name));
        
        // Serialize checkpoint (simplified - would use proper serialization)
        let serialized = serde_json::to_string(checkpoint)
            .map_err(|e| AnvilError::ComputationError(format!("Failed to serialize checkpoint: {}", e)))?;
        
        std::fs::write(&checkpoint_path, serialized)
            .map_err(|e| AnvilError::ComputationError(format!("Failed to write checkpoint: {}", e)))?;
        
        println!("Saved checkpoint: {:?}", checkpoint_path);
        Ok(())
    }
    
    pub async fn load_checkpoint(&self, name: &str) -> AnvilResult<Checkpoint> {
        let checkpoint_path = self.checkpoint_dir.join(format!("{}.ckpt", name));
        
        let serialized = std::fs::read_to_string(&checkpoint_path)
            .map_err(|e| AnvilError::ComputationError(format!("Failed to read checkpoint: {}", e)))?;
        
        let checkpoint: Checkpoint = serde_json::from_str(&serialized)
            .map_err(|e| AnvilError::ComputationError(format!("Failed to deserialize checkpoint: {}", e)))?;
        
        println!("Loaded checkpoint: {:?}", checkpoint_path);
        Ok(checkpoint)
    }
    
    pub async fn list_checkpoints(&self) -> AnvilResult<Vec<String>> {
        let mut checkpoints = Vec::new();
        
        if self.checkpoint_dir.exists() {
            let entries = std::fs::read_dir(&self.checkpoint_dir)
                .map_err(|e| AnvilError::ComputationError(format!("Failed to read checkpoint directory: {}", e)))?;
            
            for entry in entries {
                let entry = entry.map_err(|e| AnvilError::ComputationError(format!("Failed to read directory entry: {}", e)))?;
                let path = entry.path();
                
                if let Some(extension) = path.extension() {
                    if extension == "ckpt" {
                        if let Some(name) = path.file_stem() {
                            if let Some(name_str) = name.to_str() {
                                checkpoints.push(name_str.to_string());
                            }
                        }
                    }
                }
            }
        }
        
        checkpoints.sort();
        Ok(checkpoints)
    }
    
    pub async fn cleanup_old_checkpoints(&self, keep_count: usize) -> AnvilResult<()> {
        let mut checkpoints = self.list_checkpoints().await?;
        
        if checkpoints.len() <= keep_count {
            return Ok(());
        }
        
        // Sort by modification time and keep only the newest ones
        checkpoints.sort_by(|a, b| {
            let path_a = self.checkpoint_dir.join(format!("{}.ckpt", a));
            let path_b = self.checkpoint_dir.join(format!("{}.ckpt", b));
            
            let time_a = std::fs::metadata(&path_a)
                .and_then(|m| m.modified())
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
            let time_b = std::fs::metadata(&path_b)
                .and_then(|m| m.modified())
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
            
            time_b.cmp(&time_a) // Newest first
        });
        
        // Remove old checkpoints
        for checkpoint in checkpoints.iter().skip(keep_count) {
            let checkpoint_path = self.checkpoint_dir.join(format!("{}.ckpt", checkpoint));
            std::fs::remove_file(&checkpoint_path)
                .map_err(|e| AnvilError::ComputationError(format!("Failed to remove old checkpoint: {}", e)))?;
            println!("Removed old checkpoint: {:?}", checkpoint_path);
        }
        
        Ok(())
    }
    
    pub fn checkpoint_dir(&self) -> &Path {
        &self.checkpoint_dir
    }
}

// Make Checkpoint serializable (simplified implementation)
impl serde::Serialize for Checkpoint {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("Checkpoint", 6)?;
        state.serialize_field("epoch", &self.epoch)?;
        state.serialize_field("step", &self.step)?;
        state.serialize_field("loss", &self.loss)?;
        state.serialize_field("learning_rate", &self.learning_rate)?;
        state.serialize_field("optimizer_state", &self.optimizer_state)?;
        state.serialize_field("model_state", &self.model_state)?;
        state.end()
    }
}

impl<'de> serde::Deserialize<'de> for Checkpoint {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{self, Deserializer, MapAccess, Visitor};
        use std::fmt;
        
        struct CheckpointVisitor;
        
        impl<'de> Visitor<'de> for CheckpointVisitor {
            type Value = Checkpoint;
            
            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct Checkpoint")
            }
            
            fn visit_map<V>(self, mut map: V) -> Result<Checkpoint, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut epoch = None;
                let mut step = None;
                let mut loss = None;
                let mut learning_rate = None;
                let mut optimizer_state = None;
                let mut model_state = None;
                
                while let Some(key) = map.next_key()? {
                    match key {
                        "epoch" => {
                            epoch = Some(map.next_value()?);
                        }
                        "step" => {
                            step = Some(map.next_value()?);
                        }
                        "loss" => {
                            loss = Some(map.next_value()?);
                        }
                        "learning_rate" => {
                            learning_rate = Some(map.next_value()?);
                        }
                        "optimizer_state" => {
                            optimizer_state = Some(map.next_value()?);
                        }
                        "model_state" => {
                            model_state = Some(map.next_value()?);
                        }
                        _ => {
                            let _: serde::de::IgnoredAny = map.next_value()?;
                        }
                    }
                }
                
                Ok(Checkpoint {
                    epoch: epoch.ok_or_else(|| de::Error::missing_field("epoch"))?,
                    step: step.ok_or_else(|| de::Error::missing_field("step"))?,
                    loss: loss.ok_or_else(|| de::Error::missing_field("loss"))?,
                    learning_rate: learning_rate.ok_or_else(|| de::Error::missing_field("learning_rate"))?,
                    optimizer_state: optimizer_state.ok_or_else(|| de::Error::missing_field("optimizer_state"))?,
                    model_state: model_state.ok_or_else(|| de::Error::missing_field("model_state"))?,
                    metrics: TrainingMetrics::new(), // Default for now
                })
            }
        }
        
        deserializer.deserialize_struct("Checkpoint", &["epoch", "step", "loss", "learning_rate", "optimizer_state", "model_state"], CheckpointVisitor)
    }
}