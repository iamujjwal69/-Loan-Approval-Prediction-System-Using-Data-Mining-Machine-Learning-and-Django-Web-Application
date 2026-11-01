"""
Applicant Loan Application Form for Real-Time Inference.
"""

from django import forms
from src.config import CATEGORICAL_CHOICES


class ApplicantLoanForm(forms.Form):
    gender = forms.ChoiceField(
        choices=[("Male", "Male"), ("Female", "Female")],
        widget=forms.Select(attrs={"class": "form-select"}),
        label="Gender",
    )
    married = forms.ChoiceField(
        choices=[("Yes", "Yes"), ("No", "No")],
        widget=forms.Select(attrs={"class": "form-select"}),
        label="Married",
    )
    dependents = forms.ChoiceField(
        choices=[("0", "0"), ("1", "1"), ("2", "2"), ("3+", "3+ (Three or more)")],
        widget=forms.Select(attrs={"class": "form-select"}),
        label="Dependents",
    )
    education = forms.ChoiceField(
        choices=[("Graduate", "Graduate"), ("Not Graduate", "Not Graduate")],
        widget=forms.Select(attrs={"class": "form-select"}),
        label="Education Level",
    )
    self_employed = forms.ChoiceField(
        choices=[("No", "No"), ("Yes", "Yes")],
        widget=forms.Select(attrs={"class": "form-select"}),
        label="Self Employed",
    )
    applicant_income = forms.FloatField(
        min_value=0.0,
        initial=5000.0,
        widget=forms.NumberInput(attrs={"class": "form-control", "placeholder": "e.g. 5000"}),
        label="Applicant Income ($/month)",
    )
    coapplicant_income = forms.FloatField(
        min_value=0.0,
        initial=0.0,
        widget=forms.NumberInput(attrs={"class": "form-control", "placeholder": "e.g. 1500"}),
        label="Coapplicant Income ($/month)",
    )
    loan_amount = forms.FloatField(
        min_value=1.0,
        initial=130.0,
        widget=forms.NumberInput(attrs={"class": "form-control", "placeholder": "e.g. 130"}),
        label="Loan Amount (in Thousands, e.g. 130 = $130,000)",
    )
    loan_amount_term = forms.FloatField(
        min_value=12.0,
        initial=360.0,
        widget=forms.NumberInput(attrs={"class": "form-control", "placeholder": "e.g. 360"}),
        label="Loan Amount Term (in Months, 360 = 30 Years)",
    )
    credit_history = forms.ChoiceField(
        choices=[(1.0, "Meets Guidelines (1.0)"), (0.0, "Guidelines Not Met (0.0)")],
        widget=forms.Select(attrs={"class": "form-select"}),
        label="Credit History",
    )
    property_area = forms.ChoiceField(
        choices=[("Semiurban", "Semiurban"), ("Urban", "Urban"), ("Rural", "Rural")],
        widget=forms.Select(attrs={"class": "form-select"}),
        label="Property Area",
    )

    def to_applicant_dict(self):
        cd = self.cleaned_data
        return {
            "Gender": cd["gender"],
            "Married": cd["married"],
            "Dependents": str(cd["dependents"]),
            "Education": cd["education"],
            "Self_Employed": cd["self_employed"],
            "ApplicantIncome": float(cd["applicant_income"]),
            "CoapplicantIncome": float(cd["coapplicant_income"]),
            "LoanAmount": float(cd["loan_amount"]),
            "Loan_Amount_Term": float(cd["loan_amount_term"]),
            "Credit_History": float(cd["credit_history"]),
            "Property_Area": cd["property_area"],
        }
