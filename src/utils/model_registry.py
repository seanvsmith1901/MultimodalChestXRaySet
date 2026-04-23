import json
from pathlib import Path
from datetime import datetime

class ModelRegistry:
    """
    Tracks model versions and prevents overwriting better models.
    """
    
    def __init__(self, registry_path="../models/registry.json"):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry = self._load_registry()
    
    def _load_registry(self):
        """Load existing registry or create new one."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    content = f.read().strip()
                    if content:  # File is not empty
                        return json.loads(content)
                    else:
                        print("⚠️ Registry file empty, creating new one")
            except json.JSONDecodeError:
                print("⚠️ Registry file corrupted, creating new one")
        
        # Return fresh registry
        return {
            'models': {},
            'history': []
        }
    
    def _save_registry(self):
        """Save registry to disk."""
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def get_best_auc(self, model_name):
        """Get current best AUC for a model."""
        if self.registry is None:  # ✅ Safety check
            self.registry = {'models': {}, 'history': []}
        
        if model_name in self.registry.get("models", {}):
            return self.registry["models"][model_name]["auc"]
        return 0.0
    
    def get_best_info(self, model_name):
        """Get full info about best model."""
        if self.registry is None:
            return None
        return self.registry.get("models", {}).get(model_name, None)
    
    def should_save(self, model_name, new_auc, min_improvement=0.001):
        """
        Check if new model is better than current best.
        
        Returns:
            (should_save, reason, improvement)
        """
        current_best = self.get_best_auc(model_name)
        
        if new_auc > current_best + min_improvement:
            improvement = new_auc - current_best
            return True, f"Improved by +{improvement:.4f}", improvement
        elif new_auc > current_best:
            return False, f"Marginal improvement ({new_auc:.4f} vs {current_best:.4f})", new_auc - current_best
        else:
            return False, f"Worse than current best ({new_auc:.4f} vs {current_best:.4f})", new_auc - current_best
    
    def register_model(self, model_name, auc, file_path, metadata=None):
        """
        Register a new model version.
        """
        if metadata is None:
            metadata = {}
        
        # Ensure registry is valid
        if self.registry is None:
            self.registry = {'models': {}, 'history': []}
        
        timestamp = datetime.now().isoformat()
        
        # Create record
        record = {
            'model_name': model_name,
            'auc': auc,
            'file_path': str(file_path),
            'timestamp': timestamp,
            'metadata': metadata
        }
        
        # Check if this is the new best
        should_save, reason, improvement = self.should_save(model_name, auc)
        
        if should_save:
            old_best = self.registry['models'].get(model_name)
            self.registry['models'][model_name] = {
                'auc': auc,
                'file_path': str(file_path),
                'timestamp': timestamp,
                'metadata': metadata,
                'previous_best': old_best['auc'] if old_best else None
            }
            print(f"✅ Registered as NEW BEST for {model_name}")
            print(f"   AUC: {auc:.4f} (improved by +{improvement:.4f})")
        else:
            print(f"ℹ️ Not registered as best for {model_name}")
            print(f"   {reason}")
        
        # Always add to history
        if 'history' not in self.registry:
            self.registry['history'] = []
        self.registry['history'].append(record)
        self._save_registry()
        
        return should_save, reason, improvement
    
    def list_models(self):
        """Print all registered best models."""
        print("\n" + "="*60)
        print("MODEL REGISTRY - CURRENT BEST MODELS")
        print("="*60)
        
        if not self.registry or not self.registry.get('models'):
            print("No models registered yet.")
            return
        
        for name, info in self.registry['models'].items():
            print(f"\n📦 {name}")
            print(f"   AUC: {info['auc']:.4f}")
            print(f"   File: {Path(info['file_path']).name}")
            print(f"   Saved: {info['timestamp'][:10]}")
            if info.get('metadata'):
                for k, v in info['metadata'].items():
                    if v:  # Only print non-empty values
                        print(f"   {k}: {v}")