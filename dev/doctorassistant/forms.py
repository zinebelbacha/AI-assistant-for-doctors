from django import forms
from doctorassistant.models import MedicalRecord

class MedicalRecordForm(forms.ModelForm):
    class Meta:
        model = MedicalRecord
        fields = ['patient_name', 'age', 'diagnosis', 'file']

