import numpy as np
import random

def generate_unique_dataset(student_id):
    """
    Generates a unique dataset based on the student's ID.
    """
    random.seed(student_id)  # Ensures the dataset is unique for each student
    np.random.seed(student_id)
    
    # Generate random features
    X = 10 * np.random.rand(100, 2)  # 100 samples, 2 features
    noise = np.random.randn(100, 1) * random.uniform(0.5, 2.0)
    
    # Generate target values with some random coefficients
    coef = np.random.uniform(1, 5, size=(2, 1))  # 2 coefficients for 2 features
    y = X.dot(coef) + 3 + noise  # Linear relationship with noise
    
    # Save dataset to file
    np.savetxt(f"student_{student_id}_data.csv", np.hstack((X, y)), delimiter=",", 
               header="Feature1,Feature2,Target", comments="")
    print(f"Dataset for student {student_id} saved as student_{student_id}_data.csv")

# Example usage
# Replace with student-specific IDs to generate unique datasets
#student_ids = [12345, 67890, 54321, 98765]
student_ids = [1666540]
for sid in student_ids:
    generate_unique_dataset(sid)
