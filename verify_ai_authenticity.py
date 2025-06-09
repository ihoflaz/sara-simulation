#!/usr/bin/env python3
"""
AI Model Authenticity Verification Script
Tests whether the CNN model is making real predictions vs. fake logs
"""

import os
import sys
import time
import numpy as np
import torch
from typing import Dict, Tuple

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai.cnn_model import FrequencyHoppingCNN
from config import *

class AIAuthenticityTester:
    """Test the authenticity of AI model predictions"""
    
    def __init__(self):
        self.model_path = 'models/frequency_hopping_model.pth'
        self.ai_model = None
        self.model_loaded = False
        
    def load_model(self) -> bool:
        """Load the AI model"""
        try:
            if os.path.exists(self.model_path):
                self.ai_model = FrequencyHoppingCNN(input_size=30)
                checkpoint = torch.load(self.model_path, map_location='cpu')
                self.ai_model.load_state_dict(checkpoint['model_state_dict'])
                self.ai_model.eval()
                self.model_loaded = True
                print(f"✅ AI model loaded successfully from {self.model_path}")
                return True
            else:
                print(f"❌ Model file not found: {self.model_path}")
                return False
        except Exception as e:
            print(f"❌ Failed to load AI model: {str(e)}")
            return False
    
    def generate_test_features(self, scenario: str = 'test') -> np.ndarray:
        """Generate test features for all 5 bands"""
        features = []
        
        for band in range(1, 6):
            # Simulate realistic channel features for each band
            if scenario == 'jammer_band_3' and band == 3:
                # Band 3 is heavily jammed
                snr = np.random.normal(5.0, 2.0)  # Low SNR
                interference = np.random.normal(15.0, 3.0)  # High interference
                rss = np.random.normal(-85.0, 2.0)  # Weak signal
            elif scenario == 'excellent_band_1' and band == 1:
                # Band 1 has excellent conditions
                snr = np.random.normal(35.0, 2.0)  # High SNR
                interference = np.random.normal(0.5, 0.2)  # Low interference
                rss = np.random.normal(-55.0, 2.0)  # Strong signal
            else:
                # Normal conditions
                snr = np.random.normal(20.0, 5.0)
                interference = np.random.normal(3.0, 1.0)
                rss = np.random.normal(-70.0, 5.0)
            
            band_features = [
                max(0, snr),  # SNR
                max(0.1, interference),  # Interference
                rss,  # RSS
                20.0,  # Coherence time
                5.0,   # Doppler spread
                0.1    # Delay spread
            ]
            
            features.extend(band_features)
        
        return np.array(features)
    
    def test_prediction_consistency(self, num_tests: int = 100) -> Dict:
        """Test if predictions are consistent for identical inputs"""
        if not self.model_loaded:
            return {"error": "Model not loaded"}
        
        print(f"\n🧪 Testing prediction consistency with {num_tests} identical inputs...")
        
        # Create a fixed test input
        test_features = self.generate_test_features('test')
        
        predictions = []
        confidences = []
        
        for i in range(num_tests):
            with torch.no_grad():
                features_tensor = torch.FloatTensor(test_features).unsqueeze(0)
                probabilities = self.ai_model(features_tensor)
                confidence = torch.max(probabilities).item()
                predicted_band = torch.argmax(probabilities, dim=1).item() + 1
                
                predictions.append(predicted_band)
                confidences.append(confidence)
        
        unique_predictions = len(set(predictions))
        most_common_prediction = max(set(predictions), key=predictions.count)
        consistency_rate = predictions.count(most_common_prediction) / num_tests
        
        results = {
            'total_tests': num_tests,
            'unique_predictions': unique_predictions,
            'most_common_prediction': most_common_prediction,
            'consistency_rate': consistency_rate,
            'avg_confidence': np.mean(confidences),
            'std_confidence': np.std(confidences),
            'predictions_sample': predictions[:10],
            'confidences_sample': confidences[:10]
        }
        
        print(f"   📊 Results:")
        print(f"      Unique predictions: {unique_predictions}")
        print(f"      Most common prediction: Band {most_common_prediction}")
        print(f"      Consistency rate: {consistency_rate:.1%}")
        print(f"      Average confidence: {results['avg_confidence']:.4f}")
        print(f"      Confidence std dev: {results['std_confidence']:.6f}")
        
        if consistency_rate > 0.95:
            print("   ✅ Model shows high consistency - predictions are deterministic")
        elif consistency_rate > 0.8:
            print("   ⚠️  Model shows moderate consistency - may have some randomness")
        else:
            print("   ❌ Model shows low consistency - predictions are highly random")
        
        return results
    
    def test_scenario_responses(self) -> Dict:
        """Test if model responds appropriately to different scenarios"""
        if not self.model_loaded:
            return {"error": "Model not loaded"}
        
        print(f"\n🎯 Testing scenario-specific responses...")
        
        scenarios = {
            'jammer_band_3': 'Band 3 heavily jammed',
            'excellent_band_1': 'Band 1 excellent conditions',
            'normal': 'Normal conditions'
        }
        
        results = {}
        
        for scenario, description in scenarios.items():
            print(f"   🔬 Testing: {description}")
            
            scenario_predictions = []
            scenario_confidences = []
            
            # Test each scenario multiple times
            for _ in range(20):
                features = self.generate_test_features(scenario)
                
                with torch.no_grad():
                    features_tensor = torch.FloatTensor(features).unsqueeze(0)
                    probabilities = self.ai_model(features_tensor)
                    confidence = torch.max(probabilities).item()
                    predicted_band = torch.argmax(probabilities, dim=1).item() + 1
                    
                    scenario_predictions.append(predicted_band)
                    scenario_confidences.append(confidence)
            
            band_distribution = {}
            for band in range(1, 6):
                band_distribution[f'band_{band}'] = scenario_predictions.count(band)
            
            results[scenario] = {
                'predictions': scenario_predictions,
                'avg_confidence': np.mean(scenario_confidences),
                'band_distribution': band_distribution,
                'most_preferred': max(band_distribution.items(), key=lambda x: x[1])
            }
            
            print(f"      Most preferred band: {results[scenario]['most_preferred'][0].replace('band_', 'Band ')} ({results[scenario]['most_preferred'][1]}/20 times)")
            print(f"      Average confidence: {results[scenario]['avg_confidence']:.4f}")
        
        # Analyze if model behaves logically
        jammer_avoids_band_3 = results['jammer_band_3']['band_distribution']['band_3'] < 5
        excellent_prefers_band_1 = results['excellent_band_1']['band_distribution']['band_1'] > 10
        
        print(f"\n   📈 Behavioral Analysis:")
        print(f"      Avoids jammed band 3: {'✅ Yes' if jammer_avoids_band_3 else '❌ No'}")
        print(f"      Prefers excellent band 1: {'✅ Yes' if excellent_prefers_band_1 else '❌ No'}")
        
        return results
    
    def test_feature_sensitivity(self) -> Dict:
        """Test if model is sensitive to input feature changes"""
        if not self.model_loaded:
            return {"error": "Model not loaded"}
        
        print(f"\n🔧 Testing feature sensitivity...")
        
        base_features = self.generate_test_features('normal')
        
        # Get baseline prediction
        with torch.no_grad():
            features_tensor = torch.FloatTensor(base_features).unsqueeze(0)
            baseline_probs = self.ai_model(features_tensor)
            baseline_band = torch.argmax(baseline_probs, dim=1).item() + 1
        
        print(f"   Baseline prediction: Band {baseline_band}")
        
        # Test sensitivity to SNR changes in band 1
        print(f"   Testing SNR sensitivity for Band 1:")
        snr_responses = []
        
        for snr_boost in [0, 5, 10, 15, 20]:
            modified_features = base_features.copy()
            modified_features[0] += snr_boost  # Boost Band 1 SNR
            
            with torch.no_grad():
                features_tensor = torch.FloatTensor(modified_features).unsqueeze(0)
                probabilities = self.ai_model(features_tensor)
                predicted_band = torch.argmax(probabilities, dim=1).item() + 1
                band_1_prob = probabilities[0][0].item()  # Probability for Band 1
                
                snr_responses.append({
                    'snr_boost': snr_boost,
                    'predicted_band': predicted_band,
                    'band_1_probability': band_1_prob
                })
                
                print(f"      SNR +{snr_boost}dB → Band {predicted_band} (Band 1 prob: {band_1_prob:.4f})")
        
        # Check if Band 1 probability increases with SNR
        probs = [r['band_1_probability'] for r in snr_responses]
        sensitivity = np.corrcoef(range(len(probs)), probs)[0, 1]
        
        print(f"   📊 SNR sensitivity correlation: {sensitivity:.4f}")
        
        if sensitivity > 0.7:
            print("   ✅ Model is highly sensitive to SNR changes")
        elif sensitivity > 0.3:
            print("   ⚠️  Model shows moderate sensitivity to SNR changes")
        else:
            print("   ❌ Model shows low sensitivity to SNR changes")
        
        return {
            'baseline_prediction': baseline_band,
            'snr_responses': snr_responses,
            'sensitivity_correlation': sensitivity
        }
    
    def run_full_authenticity_test(self) -> Dict:
        """Run complete authenticity verification"""
        print("🔍 AI Model Authenticity Verification")
        print("=" * 50)
        
        if not self.load_model():
            return {"error": "Could not load model"}
        
        # Run all tests
        consistency_results = self.test_prediction_consistency()
        scenario_results = self.test_scenario_responses()
        sensitivity_results = self.test_feature_sensitivity()
        
        # Overall assessment
        print(f"\n🎯 OVERALL ASSESSMENT:")
        print("=" * 30)
        
        is_authentic = True
        authenticity_score = 0
        
        # Check consistency (should be high for deterministic model)
        if consistency_results.get('consistency_rate', 0) > 0.95:
            print("✅ Prediction consistency: EXCELLENT")
            authenticity_score += 25
        elif consistency_results.get('consistency_rate', 0) > 0.8:
            print("⚠️  Prediction consistency: MODERATE")
            authenticity_score += 15
        else:
            print("❌ Prediction consistency: POOR")
            is_authentic = False
        
        # Check scenario responsiveness
        jammer_test = scenario_results.get('jammer_band_3', {}).get('band_distribution', {}).get('band_3', 20) < 5
        excellent_test = scenario_results.get('excellent_band_1', {}).get('band_distribution', {}).get('band_1', 0) > 10
        
        if jammer_test and excellent_test:
            print("✅ Scenario responsiveness: EXCELLENT")
            authenticity_score += 25
        elif jammer_test or excellent_test:
            print("⚠️  Scenario responsiveness: MODERATE")
            authenticity_score += 15
        else:
            print("❌ Scenario responsiveness: POOR")
            is_authentic = False
        
        # Check feature sensitivity
        if sensitivity_results.get('sensitivity_correlation', 0) > 0.7:
            print("✅ Feature sensitivity: HIGH")
            authenticity_score += 25
        elif sensitivity_results.get('sensitivity_correlation', 0) > 0.3:
            print("⚠️  Feature sensitivity: MODERATE")
            authenticity_score += 15
        else:
            print("❌ Feature sensitivity: LOW")
            is_authentic = False
        
        # Model architecture check
        if self.ai_model is not None:
            param_count = sum(p.numel() for p in self.ai_model.parameters())
            if param_count > 50000:  # Reasonable model size
                print("✅ Model complexity: SUFFICIENT")
                authenticity_score += 25
            else:
                print("⚠️  Model complexity: LIMITED")
                authenticity_score += 15
        
        print(f"\n🏆 AUTHENTICITY SCORE: {authenticity_score}/100")
        
        if authenticity_score >= 90:
            verdict = "🟢 HIGHLY AUTHENTIC - AI model is making real decisions"
        elif authenticity_score >= 70:
            verdict = "🟡 MODERATELY AUTHENTIC - AI model works but may have limitations"
        elif authenticity_score >= 50:
            verdict = "🟠 QUESTIONABLE AUTHENTICITY - AI model shows some real behavior"
        else:
            verdict = "🔴 LOW AUTHENTICITY - AI model may be using simplified logic"
        
        print(f"🎯 VERDICT: {verdict}")
        
        return {
            'authenticity_score': authenticity_score,
            'is_authentic': is_authentic,
            'verdict': verdict,
            'consistency_results': consistency_results,
            'scenario_results': scenario_results,
            'sensitivity_results': sensitivity_results
        }

def main():
    """Main verification function"""
    tester = AIAuthenticityTester()
    results = tester.run_full_authenticity_test()
    
    # Save results for further analysis
    import json
    with open('ai_authenticity_report.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj
        
        json_results = json.loads(json.dumps(results, default=convert_numpy))
        json.dump(json_results, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: ai_authenticity_report.json")

if __name__ == "__main__":
    main()
