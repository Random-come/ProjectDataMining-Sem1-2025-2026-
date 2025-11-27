# HOW TO RUN DATA MINING PROJECT

## System Requirements
- **JDK 17** (installed and configured)
- **Java Compiler (javac)** and **Java Runtime (java)** in PATH

## Project Structure

```
code/
├── src/
│   ├── NaiveBayesModel.java          # Naive Bayes Model
│   ├── RandomForestModel.java        # Random Forest Model
│   ├── lib/
│   │   └── weka.jar                  # Weka library
│   ├── fold9_train/
│   │   └── train_90.arff             # Training data
│   └── fold1_test/
│       └── test_10.arff              # Test data
├── out/                              # Directory containing .class files after compilation
└── models/                           # Directory to save model results
    ├── NAIVEBAYES.model
    └── RANDOMFOREST.model
```

## Steps to Run the Project

### Step 1: Open Terminal/PowerShell
Open Terminal or PowerShell and navigate to the project directory:
```powershell
# If you're in the project root directory
cd GroupD_Names\code

# Or if you're already in the code directory, you can skip this step
```

### Step 2: Compile Java Code
Run the following command to compile the Java files:
```powershell
javac -cp ".;src/lib/weka.jar" -d out src\*.java
```
**Explanation:**
- `-cp ".;src/lib/weka.jar"`: Add classpath including current directory and weka.jar file
- `-d out`: Save .class files to the `out` directory
- `src\*.java`: Compile all .java files in the src directory

### Step 3: Run Naive Bayes Model
Run the following command to train and evaluate the Naive Bayes model:
```powershell
java --add-opens java.base/java.lang=ALL-UNNAMED -cp ".;out;src/lib/weka.jar" NaiveBayesModel src\fold9_train\train_90.arff src\fold1_test\test_10.arff models
java --add-opens java.base/java.lang=ALL-UNNAMED -cp ".;out;src/lib/weka.jar" NaiveBayesModel src\fold9_train\train_SMOTE-100.arff src\fold1_test\test_10.arff models
java --add-opens java.base/java.lang=ALL-UNNAMED -cp ".;out;src/lib/weka.jar" NaiveBayesModel src\fold9_train\train_SMOTE-200.arff src\fold1_test\test_10.arff models
```

**Results:**
- Model saved to: `models\NAIVEBAYES.model`
- Displays training time and accuracy

**Example output:**
```
Model saved to ...\models\NAIVEBAYES.model
Training runtime: 360 ms
Accuracy on test set: 79.40%
```

### Step 4: Run Random Forest Model
Run the following command to train and evaluate the Random Forest model:
```powershell
java --add-opens java.base/java.lang=ALL-UNNAMED -cp ".;out;src/lib/weka.jar" RandomForestModel src\fold9_train\train_90.arff src\fold1_test\test_10.arff models
java --add-opens java.base/java.lang=ALL-UNNAMED -cp ".;out;src/lib/weka.jar" RandomForestModel src\fold9_train\train_SMOTE-100.arff src\fold1_test\test_10.arff models
java --add-opens java.base/java.lang=ALL-UNNAMED -cp ".;out;src/lib/weka.jar" RandomForestModel src\fold9_train\train_SMOTE-200.arff src\fold1_test\test_10.arff models
```

**Results:**
- Model saved to: `models\RANDOMFOREST.model`
- Displays training time and accuracy

## Command Syntax

```
java --add-opens java.base/java.lang=ALL-UNNAMED -cp ".;out;src/lib/weka.jar" <ModelName> <train-file> <test-file> <output-path> 
```

**Parameters:**
- `<ModelName>`: `NaiveBayesModel` or `RandomForestModel`
- `<train-file>`: Path to ARFF training file (e.g., `src\fold9_train\train_90.arff`)
- `<test-file>`: Path to ARFF test file (e.g., `src\fold1_test\test_10.arff`)
- `<output-path>`: 
  - **Directory**: `models` (model will be saved with default name)
  - **Specific file**: `models\NAIVEBAYES.model` (full path to file)

**Note:**
- `--add-opens java.base/java.lang=ALL-UNNAMED`: Required for JDK 17 to avoid errors with Weka library

## Results

After successful execution, model files will be saved in the `models/` directory:
- `NAIVEBAYES.model`: Trained Naive Bayes model
- `RANDOMFOREST.model`: Trained Random Forest model

## Troubleshooting

### Error: "Unable to make protected final java.lang.Class..."
**Cause:** Missing `--add-opens` flag for JDK 17
**Solution:** Ensure `--add-opens java.base/java.lang=ALL-UNNAMED` is included in the java command

### Error: "File does not exist"
**Cause:** Incorrect path to ARFF file
**Solution:** Check the training and test file paths

### Error: "Could not find or load main class"
**Cause:** Code not compiled or incorrect classpath
**Solution:** 
1. Run the compilation command again: `javac -cp ".;src/lib/weka.jar" -d out src\*.java`
2. Check that .class files have been created in the `out` directory

## Verify Successful Execution

After running, you can verify:
1. The `models/` directory contains `.model` files
2. Terminal displays information:
   - Path to saved model file
   - Training time (ms)
   - Accuracy on test set (%)

## Complete Example

```powershell
# Navigate to project directory (from project root)
cd GroupD_Names\code

# Compile
javac -cp ".;src/lib/weka.jar" -d out src\*.java

# Run Naive Bayes
java --add-opens java.base/java.lang=ALL-UNNAMED -cp ".;out;src/lib/weka.jar" NaiveBayesModel src\fold9_train\train_90.arff src\fold1_test\test_10.arff models

# Run Random Forest
java --add-opens java.base/java.lang=ALL-UNNAMED -cp ".;out;src/lib/weka.jar" RandomForestModel src\fold9_train\train_90.arff src\fold1_test\test_10.arff models
```
### DISCLAIMER:
### THIS IS A INSTANT-INSTRUCTION WHICH MOST SUITABLE FOR MY SET UP, MAY NOT RUN COMFORTABLY ON SOME DEVICES  