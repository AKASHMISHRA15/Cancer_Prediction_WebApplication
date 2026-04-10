from django.db import models
from django.contrib.auth.models import User
class User(models.Model):
    username = models.CharField(max_length=100)
    email = models.EmailField()
    password = models.CharField(max_length=100)

    def __str__(self):
        return self.username


class HealthCheckup(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    age = models.IntegerField()
    gender = models.BooleanField()  
    smoking = models.BooleanField()
    yellow_fingers = models.BooleanField()
    anxiety = models.BooleanField()
    peer_pressure = models.BooleanField()
    chronic_disease = models.BooleanField()
    fatigue = models.BooleanField()
    allergy = models.BooleanField()
    wheezing = models.BooleanField()
    alcohol_consumption = models.BooleanField()
    coughing = models.BooleanField()
    shortness_of_breath = models.BooleanField()
    swallowing_difficulty = models.BooleanField()
    chest_pain = models.BooleanField()
    risk_level = models.CharField(max_length=10)  
    submitted_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.risk_level} - {self.submitted_at.strftime('%Y-%m-%d')}"


class BreastCheckup(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    age = models.IntegerField()
    menopause = models.IntegerField()
    tumor_size = models.IntegerField()
    inv_nodes = models.IntegerField()
    breast = models.IntegerField()
    metastasis = models.IntegerField()
    breast_quadrant = models.IntegerField()
    history = models.IntegerField()
    result = models.IntegerField()
    date_checked = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Breast Checkup - {self.user.username} - {self.date_checked.strftime('%Y-%m-%d')}"
