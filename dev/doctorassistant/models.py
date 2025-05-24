from django.contrib.auth import get_user_model
from django.db import models
from django.contrib.auth.models import AbstractUser,  User, Group, Permission
User = get_user_model()

class MedicalRecord(models.Model):
    patient_name = models.CharField(max_length=100)
    age = models.IntegerField()
    diagnosis = models.TextField()
    file = models.FileField(upload_to='reports/')
    image = models.ImageField(upload_to='images/')

class UserProfile(AbstractUser):
    
    profile_picture = models.ImageField(upload_to='profiles/', blank=True, null=True)
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    groups = models.ManyToManyField(Group, related_name='userprofile_groups', blank=True)
    user_permissions = models.ManyToManyField(Permission, related_name='userprofile_permissions', blank=True)

    REQUIRED_FIELDS = ['email'] 
    
    def __str__(self):
        return self.user.username
