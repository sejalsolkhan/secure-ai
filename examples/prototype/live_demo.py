import time
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import sounddevice as sd
import pickle
from colorama import Fore, Style, init
from collections import deque
import warnings
from scipy.spatial.distance import mahalanobis

# Initialize Color Output
init()
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
CAPTURE_SAMPLE_RATE = 48000
CHANNELS = 1 # Mono is safer for Pi mics
CHUNK_SIZE = 4096    
ENERGY_THRESHOLD = 1.0  # Raised to eliminate floor noise
RECORD_SECONDS = 2.0     

MODEL_SAMPLE_RATE = 32000
MODEL_PATH = "models/mn10_mydata_epoch_13.pt"
MAHA_PARAMS_PATH = "models/mahalanobis_params.pkl"

OPTIMAL_TEMP = 8.54      
MC_PASSES = 10           

# Thresholds (Defaults - will be adapted dynamically)
CONFIDENCE_THRESHOLD = 0.60       
EPISTEMIC_THRESHOLD = 0.28   # Raised from 0.25 to catch real coughs (0.26)
MAHALANOBIS_THRESHOLD = 65.0 

DEVICE_INDEX = 0

# ==========================================
# PREPROCESSING
# ==========================================
N_FFT = 1024
HOP_LENGTH = 320
N_MELS = 128
WIN_LENGTH = 800
FMIN = 0.0
FMAX = 15000

def preprocess_live_audio(audio_data, sr):
    waveform = torch.from_numpy(audio_data).float()
    
    # Safe Normalization (Prevent amplifying silence)
    max_val = torch.max(torch.abs(waveform))
    if max_val > 0.01: 
        waveform = waveform / max_val
        
    if sr != MODEL_SAMPLE_RATE:
        resampler = T.Resample(orig_freq=sr, new_freq=MODEL_SAMPLE_RATE)
        waveform = resampler(waveform)

    # Standard Pipeline
    waveform = torch.nn.functional.conv1d(waveform.unsqueeze(0).unsqueeze(0), torch.tensor([[[-.97, 1]]])).squeeze(0).squeeze(0)
    window = torch.hann_window(WIN_LENGTH, periodic=False)
    stft = torch.stft(waveform, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, center=True, normalized=False, window=window, return_complex=False)
    power_spec = (stft ** 2).sum(dim=-1)
    mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(N_MELS, N_FFT, MODEL_SAMPLE_RATE, FMIN, FMAX, vtln_low=100.0, vtln_high=-500., vtln_warp_factor=1.0)
    mel_basis = torch.nn.functional.pad(mel_basis, (0, 1), mode='constant', value=0)
    melspec = torch.matmul(mel_basis, power_spec)
    melspec = (melspec + 0.00001).log()
    melspec = (melspec + 4.5) / 5.0
    return melspec.unsqueeze(0).unsqueeze(0)

# ==========================================
# MAHALANOBIS DETECTOR
# ==========================================
class MahalanobisDetector:
    def __init__(self): 
        self.cough_mean = None
        self.inv_cov = None
        self.is_calibrated = False

    def load_offline_params(self, filepath):
        print(f"{Fore.CYAN}--- Loading Coswara Calibration ---{Style.RESET_ALL}")
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.cough_mean = data['mean']
                self.inv_cov = data['precision_matrix']
                self.is_calibrated = True
                print(f"{Fore.GREEN}✓ Loaded robust parameters from Coswara.{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}❌ Failed to load params: {e}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}⚠️  System will run without OOD filter.{Style.RESET_ALL}")

    def get_distance(self, feature_vector):
        if not self.is_calibrated: return 0.0
        return mahalanobis(feature_vector, self.cough_mean, self.inv_cov)

# ==========================================
# SYSTEM LOADER
# ==========================================
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.mn.model import get_model 

def load_system():
    print(f"{Fore.CYAN}Initializing CoughGuard...{Style.RESET_ALL}")
    model = get_model(num_classes=2, width_mult=1.0, head_type="mlp", se_dims='c')
    
    try:
        state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        if 'model_state_dict' in state_dict: state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict, strict=False)
    except:
        print(f"{Fore.RED}❌ Model file not found!{Style.RESET_ALL}")
        sys.exit(1)
    
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout): m.train()
            
    return model

# ==========================================
# MAIN RUNNER
# ==========================================
def run_demo():
    model = load_system()
    ood_detector = MahalanobisDetector()
    
    # Load Offline Params (Coswara)
    ood_detector.load_offline_params(MAHA_PARAMS_PATH) 

    # Live Loop
    print("\n" + "="*50)
    print(f"{Fore.YELLOW}  PHASE 2: LIVE GUARD ACTIVE  {Style.RESET_ALL}")
    print("="*50)
    
    triggered = False
    ring = deque(maxlen=int(RECORD_SECONDS * CAPTURE_SAMPLE_RATE))

    def callback(indata, frames, time_info, status):
        nonlocal triggered
        if status: pass
        
        # Ensure Mono
        x = indata.flatten() if CHANNELS == 1 else indata.mean(axis=1)
        ring.extend(x.tolist())
        
        vol = np.linalg.norm(x) * 10
        if vol > ENERGY_THRESHOLD and not triggered:
            print(f"{Fore.WHITE}⚡ Sound Detected (Energy: {vol:.2f}){Style.RESET_ALL}")
            triggered = True

    with sd.InputStream(callback=callback, channels=CHANNELS, samplerate=CAPTURE_SAMPLE_RATE, blocksize=CHUNK_SIZE, device=DEVICE_INDEX):
        while True:
            if triggered:
                sd.sleep(int(RECORD_SECONDS * 1000)) 
                clip = np.array(ring, dtype=np.float32)
                
                # Analyze
                print(f"{Fore.BLUE}   ▶ Analyzing...{Style.RESET_ALL}")
                pass_logits, pass_feats = [], []
                
                with torch.no_grad():
                    input_tensor = preprocess_live_audio(clip, CAPTURE_SAMPLE_RATE)
                    for _ in range(MC_PASSES):
                        # DIRECT UNPACKING
                        logits, features = model(input_tensor)
                        
                        pass_logits.append(logits.numpy()[0])
                        pass_feats.append(features.numpy().flatten())

                avg_logits = np.mean(pass_logits, axis=0)
                
                # 1. Apply Temperature Scaling
                scaled_logits = avg_logits / OPTIMAL_TEMP 
                
                # 2. Numerical Stability (LogSumExp Trick)
                exp_logits = np.exp(scaled_logits - np.max(scaled_logits)) 
                probs = exp_logits / np.sum(exp_logits)
                
                conf = probs[0] # Cough Probability
                
                epi = np.mean(np.std(pass_logits, axis=0))
                dist = ood_detector.get_distance(np.mean(pass_feats, axis=0))
                
                # Using volume from ring buffer for decision logic
                vol = np.linalg.norm(clip) / np.sqrt(len(clip)) * 10 # Approx RMS
                
                print("-" * 30)
                print(f"Confidence:  {conf:.4f}")
                print(f"Uncertainty: {epi:.4f}")
                print(f"Distance:    {dist:.1f}")
                print(f"Energy:      {vol:.2f}")
                print("-" * 30)
                
                # --- ADAPTIVE THRESHOLDS ---
                req_conf = CONFIDENCE_THRESHOLD
                max_dist = MAHALANOBIS_THRESHOLD
                
                # Strict Mode for Quiet Sounds
                if vol < 20.0:
                    req_conf = 0.70   # Must be very confident
                    max_dist = 48.0   # Must be very close to cluster
                    print(f"{Fore.YELLOW}[Adaptive] Quiet Sound - Applying Strict Standards{Style.RESET_ALL}")

                # --- DECISION GATES ---
                if epi > EPISTEMIC_THRESHOLD:
                    print(f"{Fore.MAGENTA}⚠️ REJECTED: High Uncertainty ({epi:.3f}){Style.RESET_ALL}")
                elif dist > max_dist:
                    print(f"{Fore.MAGENTA}⚠️ REJECTED: Out-of-Distribution (Dist {dist:.1f} > {max_dist}){Style.RESET_ALL}")
                elif conf < req_conf:
                    print(f"{Fore.YELLOW}⚠️ REJECTED: Low Confidence ({conf:.3f} < {req_conf}){Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}🚨 COUGH DETECTED{Style.RESET_ALL}")
                
                triggered = False
                ring.clear()
                print("\nListening...")
                sd.sleep(500) # Sleep to eliminate continuous detection

if __name__ == "__main__":
    try:
        os.environ["OPENBLAS_CORETYPE"] = "ARMV8"
        run_demo()
    except KeyboardInterrupt:
        print("\nExiting...")
