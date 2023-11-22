# SINTEC-code
- **SINTEC.py**: Contains the code for BP evaluation.

## Data
The folder provides the following data recorded with different devices:

### ECG and PPG Signals
- **ECG and PPG**: Recorded using shimmer and SINTEC devices.
    - SINTEC ECG signal: Sampled at 128 Hz.
    - SINTEC PPG signal: Sampled at 32 Hz.
    - Both SHIMMER ECG and PPG data sampled at 504.12 Hz.

### Reference Data
- **.csv file**: Contains the recorded Systolic Blood Pressure (SBP) and Diastolic Blood Pressure (DBP) measured with the reference device.

## Input Files
- **.mat files**: Contain variables 'signal' and 'timestamp' required for the BP evaluation process.

- # Instructions for Code Execution

Once you run the `SINTEC.py` code, follow these steps:

## User Input Required
- **ECG File Name**: Provide the filename for the ECG signal.
- **PPG File Name**: Provide the filename for the PPG signal.
- **CSV File Name**: Specify the filename for the .csv file containing SBP and DBP data.
- **Signal Sampling Frequencies**: Provide the sampling frequencies.


## Process Steps
1. **Filtered ECG Visualization**:
   - Upon running the code, the filtered ECG signal will be visualized.
   - User Interaction: Select the appropriate threshold (e.g., 0.1 for SINTEC ECG).

2. **PPG Visualization**:
   - The PPG signal will be displayed for user assessment.
   - User Interaction: Choose the suitable threshold (e.g., 500 for SINTEC PPG).

3. **Blood Pressure (BP) Estimation**:
   - The figure containing BP values will be generated.
   - Statistical Information:
     - Mean Absolute Error (MAE) and Standard Deviation (SD) of SBP and DBP predictions will be provided.


