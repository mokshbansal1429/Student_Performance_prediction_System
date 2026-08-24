# 🎓 EduPredict AI

## AI-Based Student Performance Prediction System

EduPredict AI is a machine learning-based system that predicts a student's academic performance using various factors such as attendance, study hours, assignment scores, test marks, previous academic performance, and other relevant parameters.

The main goal of this project is not only to predict student performance but also to identify students who may be at academic risk and provide personalized recommendations to help them improve.

---

## 🎯 Project Objective

The objective of EduPredict AI is to:

- Predict a student's expected academic performance.
- Identify students who are performing well or are at risk.
- Analyze the factors affecting student performance.
- Provide personalized improvement suggestions.
- Help teachers identify students who need additional attention.
- Present student performance through an easy-to-use dashboard.

---

## 🚀 Key Features

### 1. Student Performance Prediction

The system uses a Machine Learning model to predict the expected final marks or percentage of a student.

### 2. Performance Classification

Students can be classified into categories such as:

- Excellent
- Good
- Average
- At Risk

### 3. Risk Detection

The system identifies students who may be at risk of poor academic performance based on their current academic data.

### 4. AI-Based Recommendations

Based on the student's weak areas, the system provides personalized suggestions such as:

- Improve attendance.
- Increase study hours.
- Practice more tests.
- Improve assignment performance.
- Focus on weak subjects.

### 5. Student Dashboard

Students can view:

- Attendance
- Study hours
- Assignment performance
- Test scores
- Previous marks
- Predicted performance
- Performance status
- AI recommendations

### 6. Teacher Dashboard

Teachers can view overall class performance and identify students who require additional attention.

---

## 🧠 Machine Learning

The project will use Machine Learning to analyze historical student data and learn patterns between student characteristics and their final academic performance.

### Initial Model

The initial model will use:

**Random Forest Regression**

The model will predict the student's expected final marks/percentage.

Later, classification models can also be explored for identifying performance categories and academic risk.

---

## 📊 Input Features

The model may use features such as:

| Feature | Description |
|---|---|
| Attendance | Student attendance percentage |
| Study Hours | Average daily study hours |
| Assignment Score | Assignment performance |
| Quiz/Test Score | Average test performance |
| Previous Marks | Previous academic performance |
| Internal Marks | Internal examination marks |
| Backlogs | Number of previous backlogs |

### Target

The primary target will be:

**Final Marks / Final Percentage**

---

## 🛠️ Technology Stack

### Frontend
- HTML
- CSS
- JavaScript

### Backend
- Python
- Flask

### Machine Learning
- Pandas
- NumPy
- Scikit-learn

### Database
- MySQL

### Development Tool
- Visual Studio Code

---

## 🏗️ System Architecture

```text
                Student Data
                     ↓
              Data Preprocessing
                     ↓
              Machine Learning
                     ↓
              Performance Prediction
                     ↓
          ┌──────────┴──────────┐
          ↓                     ↓
   Performance Status      Risk Detection
          ↓                     ↓
          └──────────┬──────────┘
                     ↓
           AI Recommendations
                     ↓
               Dashboard
