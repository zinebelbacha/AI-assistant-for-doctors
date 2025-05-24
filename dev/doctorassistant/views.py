from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from django.contrib.auth import login, authenticate
from django.contrib.auth.decorators import login_required
from doctorassistant.forms import MedicalRecordForm
from django.contrib.auth import get_user_model
from django.urls import reverse
from django.contrib import messages
from doctorassistant.models import UserProfile
from django.core.files.base import ContentFile
from django.contrib.auth.models import User
import base64
from django.contrib.auth.views import LoginView
import os
from django.utils.decorators import method_decorator
from django.shortcuts import redirect
from django.urls import reverse_lazy
from django.http import JsonResponse
from django.views import View
from django.conf import settings
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from doctorassistant.model_loader import get_ocr_model, get_pnortub_model
from django.core.files.storage import default_storage
from PIL import Image
import fitz  # PyMuPDF
import shutil  # Pour nettoyer les fichiers temporaires
from django.contrib.auth.forms import UserCreationForm
from django.views.generic.edit import FormView
from django.http import HttpResponseRedirect
from deepface import DeepFace
from django.contrib.auth import get_user_model
import cv2
from io import BytesIO

@method_decorator(login_required, name='dispatch')
class PredictView(View):
    def get(self, request, *args, **kwargs):
        return render(request, 'prediction_form.html')
    @staticmethod
    def convert_pdf_to_images(pdf_path, output_folder):
        pdf_document = fitz.open(pdf_path)
        image_paths = []
        try:
            for page_num in range(pdf_document.page_count):
                page = pdf_document.load_page(page_num)
                pix = page.get_pixmap()
                image_path = os.path.join(output_folder, f"page_{page_num + 1}.png")
                pix.save(image_path)
                image_paths.append(image_path)
        finally:
            pdf_document.close()  
        return image_paths

    def post(self, request, *args, **kwargs):
        # Récupération des données du formulaire
        patient_name = request.POST.get("patient_name")
        age = request.POST.get("age")
        file = request.FILES.get("file")

        # Validation des champs
        if not patient_name or not age or not file:
            return JsonResponse({"error": "All fields are required"}, status=400)

        # Sauvegarde temporaire du fichier
        file_path = os.path.join(settings.MEDIA_ROOT, file.name)
        with open(file_path, "wb") as f:
            for chunk in file.chunks():
                f.write(chunk)

        try:
            if file.name.lower().endswith(".pdf"):
                # Créez un dossier temporaire pour les images
                temp_folder = os.path.join(settings.MEDIA_ROOT, "temp_images")
                if not os.path.exists(temp_folder):
                    os.makedirs(temp_folder)

                # Convertir le PDF en images
                image_paths =PredictView.convert_pdf_to_images(file_path, temp_folder)

                # Vérifiez si des images ont été générées
                if not image_paths:
                    return JsonResponse({"error": "No images generated from PDF"}, status=400)

                predictions = []
                model = get_ocr_model()

                for i, image_path in enumerate(image_paths):
                    try:
                        # Prétraiter l'image pour le modèle OCR
                        img = load_img(image_path, target_size=(224, 224))
                        input_arr = img_to_array(img) / 255.0
                        input_arr = input_arr.reshape((1, *input_arr.shape))

                        # Effectuer la prédiction
                        result = model.predict(input_arr)
                        
    
                        # Obtenir l'index de la classe avec la probabilité la plus élevée
                        class_index = result.argmax()  # Indice de la classe avec la probabilité maximale
                        
                        # Liste des étiquettes des classes
                        class_labels = ["Normal", "Pneumonie", "Tuberculose"]
                        predicted_class=class_labels[class_index]
                        
                    
                     
                        predictions.append(f"Page {i + 1}: {predicted_class}")
                    except Exception as e:
                        predictions.append(f"Error processing page {i + 1}: {str(e)}")
                    if os.path.exists(image_path):
                        os.remove(image_path)

                # Renvoyer les prédictions pour toutes les pages
                prediction = "\n".join(predictions)

                # Supprimer les fichiers temporaires
                shutil.rmtree(temp_folder)

            elif file.name.lower().endswith(("jpg", "jpeg", "png")):
                # Si c'est une image médicale
                model = get_pnortub_model()
                image = load_img(file_path, target_size=(224, 224))
                input_arr = img_to_array(image) / 255.0
                input_arr = input_arr.reshape((1, *input_arr.shape))
                predictions = model.predict(input_arr)
                class_index = predictions.argmax()
                class_labels = ["Normal", "Pneumonie", "Tuberculose"]
                prediction = class_labels[class_index]
            else:
                return JsonResponse({"error": "Unsupported file type"}, status=400)

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
       
        return JsonResponse({
            "patient_name": patient_name,
            "age": age,
            "prediction": prediction
        })
User = get_user_model()
def custom_login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        captured_image = request.POST.get('captured_image')
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            # Sauvegarder l'image capturée
            if captured_image:
                try:
                    format, imgstr = captured_image.split(';base64,')
                    ext = format.split('/')[-1]
                    data = ContentFile(base64.b64decode(imgstr), name=f"{username}.{ext}")
                    
                    # Associez l'image à l'utilisateur via un profil
                    user_profile, created = UserProfile.objects.get_or_create(user=user)
                    user_profile.profile_picture.save(f"{username}_face.{ext}", data)
                except Exception as e:
                    return JsonResponse({'error': f"Failed to save captured image: {str(e)}"}, status=500)
            
            
            login(request, user)
            return JsonResponse({'message': 'Connexion réussie !'})
        else:
            return JsonResponse({'error': 'Nom d’utilisateur ou mot de passe incorrect.'}, status=400)
    return render(request, 'login_face.html')


def register_with_face(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        email = request.POST['email']
        profile_picture = request.FILES.get('profile_picture')

        # Vérifier si le nom d'utilisateur existe déjà
        if User.objects.filter(username=username).exists():
            messages.error(request, "Ce nom d'utilisateur est déjà pris.")
            return redirect('signup_face')
        
        # Créer un utilisateur
        user = User.objects.create_user(username=username, password=password, email=email)

        # Créer un profil utilisateur avec la photo de profil (si elle est fournie)
        if profile_picture:
            profile = UserProfile.objects.create(user=user, profile_picture=profile_picture)
            profile.save()

        messages.success(request, "Inscription réussie.")
        return redirect('login_face')

    return render(request, 'signup-face.html')

def index_view(request):
    return render(request, 'index.html')