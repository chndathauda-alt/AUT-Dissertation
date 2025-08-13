import numpy as np
import pandas as pd
import trimesh
import os
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

class FruitAnalysisSystem:
    def __init__(self, models_path="C:\\Users\\ssp4755\\Downloads\\3D models\\fruits"):
        """
        Initialize the Fruit Analysis System
        
        Args:
            models_path (str): Path to the directory containing 3D fruit models
        """
        self.models_path = models_path
        self.fruit_models = {}
        self.trained_models = {}
        self.scalers = {}
        self.feature_extractors = {}
        self.training_data = None
        
        # Nutritional features to predict
        self.nutrition_features = [
            'calories', 'carbs_g', 'fiber_g', 'sugar_g', 
            'protein_g', 'fat_g', 'vitamin_c_mg'
        ]
        
        # Geometric features to extract
        self.geometric_features = [
            'volume_ml', 'weight_g', 'surface_area_cm2', 
            'aspect_ratio', 'sphericity'
        ]
        
    def load_training_data(self, csv_path):
        """Load and prepare training data from CSV"""
        print("Loading training data...")
        self.training_data = pd.read_csv(csv_path)
        print(f"Loaded {len(self.training_data)} training samples")
        print(f"Fruit types: {self.training_data['fruit_name'].unique()}")
        return self.training_data
    
    def extract_3d_features(self, mesh):
        """
        Extract geometric features from a 3D mesh
        
        Args:
            mesh: Trimesh object
            
        Returns:
            dict: Dictionary of extracted features
        """
        features = {}
        
        # Volume (convert to ml, assuming 1 unit³ = 1 ml)
        features['volume_ml'] = mesh.volume
        
        # Surface area (convert to cm²)
        features['surface_area_cm2'] = mesh.area
        
        # Bounding box dimensions
        bounds = mesh.bounds
        dimensions = bounds[1] - bounds[0]
        
        # Aspect ratio (length/width ratio)
        sorted_dims = sorted(dimensions, reverse=True)
        features['aspect_ratio'] = sorted_dims[0] / sorted_dims[1] if sorted_dims[1] > 0 else 1.0
        
        # Sphericity calculation
        # Sphericity = (π^(1/3) * (6V)^(2/3)) / A
        volume = mesh.volume
        area = mesh.area
        if area > 0 and volume > 0:
            features['sphericity'] = (np.pi**(1/3) * (6 * volume)**(2/3)) / area
        else:
            features['sphericity'] = 0
            
        # Weight estimation (assuming density varies by fruit type)
        # This is a simplified estimation
        features['weight_g'] = volume * 0.8  # Average fruit density approximation
        
        # Additional geometric features
        features['convex_hull_volume'] = mesh.convex_hull.volume
        features['convexity'] = volume / mesh.convex_hull.volume if mesh.convex_hull.volume > 0 else 0
        
        # Centroid
        centroid = mesh.centroid
        features['centroid_x'] = centroid[0]
        features['centroid_y'] = centroid[1]
        features['centroid_z'] = centroid[2]
        
        # Moments of inertia
        inertia = mesh.moment_inertia
        features['inertia_xx'] = inertia[0, 0]
        features['inertia_yy'] = inertia[1, 1]
        features['inertia_zz'] = inertia[2, 2]
        
        return features
    
    def load_3d_models(self):
        """Load all 3D models from the specified directory"""
        print(f"Loading 3D models from: {self.models_path}")
        
        if not os.path.exists(self.models_path):
            print(f"Error: Directory {self.models_path} does not exist!")
            return False
            
        model_files = [f for f in os.listdir(self.models_path) if f.endswith('.obj')]
        
        if not model_files:
            print("No .obj files found in the directory!")
            return False
            
        print(f"Found {len(model_files)} model files")
        
        for filename in model_files:
            try:
                filepath = os.path.join(self.models_path, filename)
                mesh = trimesh.load(filepath)
                
                # Extract fruit name from filename
                fruit_name = filename.replace('.obj', '').lower()
                
                self.fruit_models[fruit_name] = {
                    'mesh': mesh,
                    'filepath': filepath,
                    'features': self.extract_3d_features(mesh)
                }
                
                print(f"Loaded: {fruit_name}")
                
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
                
        return len(self.fruit_models) > 0
    
    def prepare_training_features(self):
        """Prepare features for training machine learning models"""
        if self.training_data is None:
            print("Please load training data first!")
            return None
            
        # Extract features for each fruit type
        feature_data = []
        
        for fruit_type in self.training_data['fruit_name'].unique():
            fruit_data = self.training_data[self.training_data['fruit_name'] == fruit_type]
            
            # Create feature vectors
            for _, row in fruit_data.iterrows():
                features = {
                    'fruit_type': fruit_type,
                    'volume_ml': row['volume_ml'],
                    'weight_g': row['weight_g'],
                    'surface_area_cm2': row['surface_area_cm2'],
                    'aspect_ratio': row['aspect_ratio'],
                    'sphericity': row['sphericity'],
                }
                
                # Add nutritional targets
                for nutrient in self.nutrition_features:
                    features[nutrient] = row[nutrient]
                    
                feature_data.append(features)
        
        return pd.DataFrame(feature_data)
    
    def train_models(self):
        """Train machine learning models for each nutritional feature"""
        print("Training prediction models...")
        
        # Prepare training features
        feature_df = self.prepare_training_features()
        if feature_df is None:
            return False
        
        # Prepare feature matrix
        X_features = ['volume_ml', 'weight_g', 'surface_area_cm2', 'aspect_ratio', 'sphericity']
        X = feature_df[X_features].values
        
        # Train separate models for each nutritional feature
        for nutrient in self.nutrition_features:
            print(f"Training model for {nutrient}...")
            
            y = feature_df[nutrient].values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train Random Forest model
            model = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=5
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"  {nutrient} - MAE: {mae:.3f}, R²: {r2:.3f}")
            
            # Store model and scaler
            self.trained_models[nutrient] = model
            self.scalers[nutrient] = scaler
        
        print("Model training completed!")
        return True
    
    def save_models(self, save_path="trained_models"):
        """Save trained models to disk"""
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        # Save models
        for nutrient in self.nutrition_features:
            if nutrient in self.trained_models:
                model_file = os.path.join(save_path, f"{nutrient}_model.pkl")
                scaler_file = os.path.join(save_path, f"{nutrient}_scaler.pkl")
                
                with open(model_file, 'wb') as f:
                    pickle.dump(self.trained_models[nutrient], f)
                    
                with open(scaler_file, 'wb') as f:
                    pickle.dump(self.scalers[nutrient], f)
        
        print(f"Models saved to {save_path}")
    
    def load_models(self, load_path="trained_models"):
        """Load trained models from disk"""
        for nutrient in self.nutrition_features:
            model_file = os.path.join(load_path, f"{nutrient}_model.pkl")
            scaler_file = os.path.join(load_path, f"{nutrient}_scaler.pkl")
            
            if os.path.exists(model_file) and os.path.exists(scaler_file):
                with open(model_file, 'rb') as f:
                    self.trained_models[nutrient] = pickle.load(f)
                    
                with open(scaler_file, 'rb') as f:
                    self.scalers[nutrient] = pickle.load(f)
        
        print("Models loaded successfully!")
    
    def analyze_new_fruit(self, obj_filepath):
        """
        Analyze a new 3D fruit model and predict its nutritional content
        
        Args:
            obj_filepath (str): Path to the .obj file to analyze
            
        Returns:
            dict: Analysis results including volume and nutritional predictions
        """
        try:
            # Load the 3D model
            mesh = trimesh.load(obj_filepath)
            print(f"Loaded 3D model: {obj_filepath}")
            
            # Extract geometric features
            features = self.extract_3d_features(mesh)
            
            # Prepare feature vector for prediction
            feature_vector = np.array([
                features['volume_ml'],
                features['weight_g'],
                features['surface_area_cm2'],
                features['aspect_ratio'],
                features['sphericity']
            ]).reshape(1, -1)
            
            # Make predictions for each nutritional feature
            predictions = {'geometric_features': features}
            
            for nutrient in self.nutrition_features:
                if nutrient in self.trained_models:
                    # Scale features
                    scaled_features = self.scalers[nutrient].transform(feature_vector)
                    
                    # Predict
                    prediction = self.trained_models[nutrient].predict(scaled_features)[0]
                    predictions[nutrient] = max(0, prediction)  # Ensure non-negative values
            
            return predictions
            
        except Exception as e:
            print(f"Error analyzing fruit: {str(e)}")
            return None
    
    def display_results(self, results, filename):
        """Display analysis results in a formatted way"""
        if results is None:
            print("No results to display")
            return
            
        print(f"\n{'='*60}")
        print(f"FRUIT ANALYSIS RESULTS - {filename}")
        print(f"{'='*60}")
        
        print(f"\n🔍 GEOMETRIC MEASUREMENTS:")
        print(f"  Volume:           {results['geometric_features']['volume_ml']:.2f} ml")
        print(f"  Weight (est.):    {results['geometric_features']['weight_g']:.2f} g")
        print(f"  Surface Area:     {results['geometric_features']['surface_area_cm2']:.2f} cm²")
        print(f"  Aspect Ratio:     {results['geometric_features']['aspect_ratio']:.3f}")
        print(f"  Sphericity:       {results['geometric_features']['sphericity']:.3f}")
        print(f"  Convexity:        {results['geometric_features']['convexity']:.3f}")
        
        print(f"\n PREDICTED NUTRITIONAL CONTENT:")
        print(f"  Calories:         {results.get('calories', 0):.1f} kcal")
        print(f"  Carbohydrates:    {results.get('carbs_g', 0):.2f} g")
        print(f"  Fiber:            {results.get('fiber_g', 0):.2f} g")
        print(f"  Sugar:            {results.get('sugar_g', 0):.2f} g")
        print(f"  Protein:          {results.get('protein_g', 0):.2f} g")
        print(f"  Fat:              {results.get('fat_g', 0):.2f} g")
        print(f"  Vitamin C:        {results.get('vitamin_c_mg', 0):.1f} mg")
        
        print(f"\n{'='*60}")
    
    def create_analysis_report(self, results, filename, save_path="analysis_reports"):
        """Create a detailed analysis report"""
        if results is None:
            return
            
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        report_file = os.path.join(save_path, f"{filename}_analysis_report.txt")
        
        with open(report_file, 'w') as f:
            f.write("FRUIT 3D MODEL ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"File: {filename}\n")
            f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("GEOMETRIC MEASUREMENTS:\n")
            f.write("-" * 25 + "\n")
            for key, value in results['geometric_features'].items():
                f.write(f"{key}: {value:.4f}\n")
            
            f.write("\nNUTRITIONAL PREDICTIONS:\n")
            f.write("-" * 25 + "\n")
            for nutrient in self.nutrition_features:
                if nutrient in results:
                    f.write(f"{nutrient}: {results[nutrient]:.4f}\n")
        
        print(f"Detailed report saved to: {report_file}")

def main():
    """Main function to demonstrate the system"""
    print(" FRUIT 3D MODEL ANALYSIS SYSTEM")
    print("=" * 50)
    
    # Initialize the system
    system = FruitAnalysisSystem()
    
    # Load training data
    csv_path = input("Enter path to training CSV file (or press Enter for 'paste.txt'): ").strip()
    if not csv_path:
        csv_path = "paste.txt"  # Default to the provided file
    
    try:
        system.load_training_data(csv_path)
    except Exception as e:
        print(f"Error loading training data: {e}")
        return
    
    # Load and train models
    print("\n1. Loading 3D models...")
    if system.load_3d_models():
        print(" 3D models loaded successfully")
    else:
        print(" Failed to load 3D models - continuing with training only")
    
    print("\n2. Training prediction models...")
    if system.train_models():
        print(" Models trained successfully")
        
        # Save models
        system.save_models()
        print(" Models saved to disk")
    else:
        print(" Model training failed")
        return
    
    # Interactive analysis loop
    print("\n" + "="*60)
    print("READY FOR ANALYSIS!")
    print("You can now upload new .obj files for analysis")
    print("="*60)
    
    while True:
        print("\nOptions:")
        print("1. Analyze a new 3D fruit model")
        print("2. Exit")
        
        choice = input("\nEnter your choice (1 or 2): ").strip()
        
        if choice == '1':
            obj_path = input("Enter path to .obj file: ").strip()
            
            if not obj_path or not os.path.exists(obj_path):
                print(" File not found!")
                continue
                
            print(f"\n Analyzing {obj_path}...")
            results = system.analyze_new_fruit(obj_path)
            
            if results:
                # Display results
                filename = os.path.basename(obj_path)
                system.display_results(results, filename)
                
                # Create detailed report
                system.create_analysis_report(results, filename.replace('.obj', ''))
                
                print(" Analysis completed!")
            else:
                print(" Analysis failed!")
                
        elif choice == '2':
            print(" Goodbye!")
            break
        else:
            print(" Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()