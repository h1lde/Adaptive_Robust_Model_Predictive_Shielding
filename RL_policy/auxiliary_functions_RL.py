from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class TD3SchedulerCallback(BaseCallback):
    """
    Callback for scheduling learning rate and action noise in TD3 algorithm.
    
    Parameters:
        lr_start: Initial learning rate
        lr_end: Final learning rate
        noise_start: Initial action noise standard deviation
        noise_end: Final action noise standard deviation
        max_timesteps: Total timesteps for the complete annealing
        warmup_steps: Number of timesteps to maintain initial values before annealing
        annealing_type: Type of annealing schedule ('linear', 'exponential', or 'piecewise')
        verbose: Verbosity level
    """
    
    def __init__(
        self,
        lr_start: float = 1e-4,
        lr_end: float = 1e-5,
        noise_start: float = 1e-1,
        noise_end: float = 1e-2,
        max_timesteps: int = 500000,
        warmup_steps: int = 10000,
        annealing_type: str = 'exponential',   # try 'linear', 'exponential', or 'piecewise'
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.lr_start = lr_start
        self.lr_end = lr_end
        self.noise_start = noise_start
        self.noise_end = noise_end
        self.max_timesteps = max_timesteps
        self.warmup_steps = warmup_steps
        self.annealing_type = annealing_type
        
    def _init_callback(self) -> None:
        # Store original learning rate for reference
        self.orig_lr = self.model.learning_rate
        # Make sure we have access to the TD3 noise parameters
        assert hasattr(self.model, "actor"), "Model does not have actor attribute, is it a TD3 model?"
        assert hasattr(self.model, "action_noise"), "Model does not have action_noise attribute"
        
        # Store original action noise std for reference
        self.orig_noise_std = self.model.action_noise._sigma
        
        # Performance monitoring
        self.episode_rewards = []
        self.current_episode_reward = 0
        
    def _compute_new_value(self, start_val, end_val, progress):
        """Compute new value based on annealing schedule"""
        if progress < self.warmup_steps / self.max_timesteps:
            # During warmup period, maintain the start value
            return start_val
            
        # Adjust progress to account for warmup
        adjusted_progress = (progress * self.max_timesteps - self.warmup_steps) / (self.max_timesteps - self.warmup_steps)
        adjusted_progress = max(0, min(1, adjusted_progress))  # Clip between 0 and 1
        
        if self.annealing_type == 'linear':
            # Linear annealing
            return start_val + (end_val - start_val) * adjusted_progress
        elif self.annealing_type == 'exponential':
            # Exponential annealing
            return start_val * (end_val / start_val) ** adjusted_progress
        elif self.annealing_type == 'piecewise':
            # Piecewise annealing (faster decay at the beginning)
            if adjusted_progress < 0.3:
                # First 30%: faster decay (50% of the way)
                sub_progress = adjusted_progress / 0.3
                return start_val + (start_val + (end_val - start_val) * 0.5 - start_val) * sub_progress
            else:
                # Remaining 70%: slower decay (remaining 50%)
                sub_progress = (adjusted_progress - 0.3) / 0.7
                mid_val = start_val + (end_val - start_val) * 0.5
                return mid_val + (end_val - mid_val) * sub_progress
        else:
            raise ValueError(f"Unknown annealing type: {self.annealing_type}")
    
    def _on_step(self) -> bool:
        # Calculate progress (0 to 1)
        progress = min(1.0, self.num_timesteps / self.max_timesteps)
        
        # Update learning rate
        new_lr = self._compute_new_value(self.lr_start, self.lr_end, progress)
        self.model.learning_rate = new_lr
        
        # Update action noise
        new_noise = self._compute_new_value(self.noise_start, self.noise_end, progress)
        self.model.action_noise._sigma = new_noise
        
        # If we've exceeded max_timesteps, ensure we maintain the final values
        if self.num_timesteps > self.max_timesteps:
            self.model.learning_rate = self.lr_end
            self.model.action_noise._sigma = self.noise_end
        
        # Track episode rewards for monitoring
        if self.model.get_env() is not None:
            info = self.locals.get("infos")[0]
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                
                # Log current values every 10 episodes
                if len(self.episode_rewards) % 10 == 0 and self.verbose > 0:
                    avg_reward = np.mean(self.episode_rewards[-10:])
                    print(f"Timestep: {self.num_timesteps}/{self.max_timesteps} | "
                          f"LR: {new_lr:.6f} | Noise: {new_noise:.6f} | "
                          f"Avg Reward (10 ep): {avg_reward:.2f}")
        
        return True

    def get_lr(self):
        """Get current learning rate"""
        return self.model.learning_rate
        
    def get_noise(self):
        """Get current action noise std"""
        return self.model.action_noise._sigma
