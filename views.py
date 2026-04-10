from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import HttpResponse
from .models import User,HealthCheckup, BreastCheckup
from django.utils import timezone
from django.contrib.auth.decorators import login_required
import joblib
import numpy as np


def info(request):
    return render(request, "info.html")

def login_view(request):
    if request.method == "POST":
        username = request.POST.get("username")
        email = request.POST.get("email")
        password = request.POST.get("password")

        user, created = User.objects.get_or_create(username=username, email=email)
        if created or user.password == password:
            user.password = password
            user.save()
            request.session["user_id"] = user.id
            return redirect("info")
        else:
            messages.error(request, "Invalid credentials")
    return render(request, "login.html")

def logout_view(request):
    request.session.flush()
    return redirect("info")


def profile_view(request):
    user_id = request.session.get('user_id')

    if not user_id:
        return redirect('login')

    try:
        user = User.objects.get(id=user_id)
    except User.DoesNotExist:
        return redirect('logout')

    health_checkups = HealthCheckup.objects.filter(user=user).order_by('-submitted_at')
    breast_checkups = BreastCheckup.objects.filter(user=user).order_by('-date_checked')


    return render(request, 'profile.html', {
        'user': user,
        'health_checkups': health_checkups,
        'breast_checkups': breast_checkups,
    })


def lung_cancer_checkup_view(request):
    user_id = request.session.get("user_id")
    try:
        user = User.objects.get(id=user_id)
    except User.DoesNotExist:
        request.session.flush()
        return redirect("login")
    

    model = joblib.load('c:/Users/AKASH MISHRA/OneDrive/Desktop/SKILLX/myapp/ml_model/lung_cancer_model.pkl')
    
     
    if request.method == "POST":
        form_data = {
            "age": request.POST.get("age", ""),
            "gender": request.POST.get("gender", ""),
            "smoking": request.POST.get("smoking", ""),
            "yellow_fingers": request.POST.get("yellow_fingers", ""),
            "anxiety": request.POST.get("anxiety", ""),
            "peer_pressure": request.POST.get("peer_pressure", ""),
            "chronic_disease": request.POST.get("chronic_disease", ""),
            "fatigue": request.POST.get("fatigue", ""),
            "allergy": request.POST.get("allergy", ""),
            "wheezing": request.POST.get("wheezing", ""),
            "alcohol": request.POST.get("alcohol", ""),
            "coughing": request.POST.get("coughing", ""),
            "shortness_of_breath": request.POST.get("shortness_of_breath", ""),
            "swallowing_difficulty": request.POST.get("swallowing_difficulty", ""),
            "chest_pain": request.POST.get("chest_pain", ""),
        }

        bool_map = lambda x: x.lower() == "yes"
        yes_count = sum([
            bool_map(form_data["smoking"]),
            bool_map(form_data["yellow_fingers"]),
            bool_map(form_data["anxiety"]),
            bool_map(form_data["peer_pressure"]),
            bool_map(form_data["chronic_disease"]),
            bool_map(form_data["fatigue"]),
            bool_map(form_data["allergy"]),
            bool_map(form_data["wheezing"]),
            bool_map(form_data["alcohol"]),
            bool_map(form_data["coughing"]),
            bool_map(form_data["shortness_of_breath"]),
            bool_map(form_data["swallowing_difficulty"]),
            bool_map(form_data["chest_pain"]),
        ])
        features = [
            int(form_data["age"]),
            1 if form_data["gender"] == "female" else 2,
            2 if bool_map(form_data["smoking"]) else 1,
            2 if bool_map(form_data["yellow_fingers"]) else 1,
            2 if bool_map(form_data["anxiety"]) else 1,
            2 if bool_map(form_data["peer_pressure"]) else 1,
            2 if bool_map(form_data["chronic_disease"]) else 1,
            2 if bool_map(form_data["fatigue"]) else 1,
            2 if bool_map(form_data["allergy"]) else 1,
            2 if bool_map(form_data["wheezing"]) else 1,
            2 if bool_map(form_data["alcohol"]) else 1,
            2 if bool_map(form_data["coughing"]) else 1,
            2 if bool_map(form_data["shortness_of_breath"]) else 1,
            2 if bool_map(form_data["swallowing_difficulty"]) else 1,
            2 if bool_map(form_data["chest_pain"]) else 1,
        ]

        if yes_count == 2:
            risk_level = "Low"
        else:
            prediction = model.predict(np.array(features).reshape(1, -1))
            risk_level = "High" if prediction == 2 else "Low"

        HealthCheckup.objects.create(
            user=user,
            age=form_data["age"],
            gender=form_data["gender"] == "male",
            smoking=bool_map(form_data["smoking"]),
            yellow_fingers=bool_map(form_data["yellow_fingers"]),
            anxiety=bool_map(form_data["anxiety"]),
            peer_pressure=bool_map(form_data["peer_pressure"]),
            chronic_disease=bool_map(form_data["chronic_disease"]),
            fatigue=bool_map(form_data["fatigue"]),
            allergy=bool_map(form_data["allergy"]),
            wheezing=bool_map(form_data["wheezing"]),
            alcohol_consumption=bool_map(form_data["alcohol"]),
            coughing=bool_map(form_data["coughing"]),
            shortness_of_breath=bool_map(form_data["shortness_of_breath"]),
            swallowing_difficulty=bool_map(form_data["swallowing_difficulty"]),
            chest_pain=bool_map(form_data["chest_pain"]),
            risk_level=risk_level,
        )

        return render(request, "results.html", {"form_data": form_data, "risk_level": risk_level})
    return render(request, "lung_cancer_checkup.html")


def breast_cancer_checkup(request):
    user_id = request.session.get('user_id')
    if not user_id:
        return redirect('login')

    try:
        user = User.objects.get(id=user_id)
    except User.DoesNotExist:
        request.session.flush()
        return redirect("login")

    breast_model = joblib.load('c:/Users/AKASH MISHRA/OneDrive/Desktop/SKILLX/myapp/ml_model/breast-cancer-model.pkl')
    scaler = joblib.load('c:/Users/AKASH MISHRA/OneDrive/Desktop/SKILLX/myapp/ml_model/breast-scaler.pkl')

    if request.method == "POST":
        form_data = request.POST

        try:
            age = int(form_data["age"])
            menopause = int(form_data["menopause"])
            tumor_size = int(form_data["tumor_size"])
            inv_nodes = int(form_data["inv_nodes"])
            breast = int(form_data["breast"])
            metastasis = int(form_data["metastasis"])
            breast_quadrant = int(form_data["breast_quadrant"])
            history = int(form_data["history"])

            input_data = [age, menopause, tumor_size, inv_nodes, breast, metastasis, breast_quadrant, history]
            scaled_input = scaler.transform([input_data])

            prediction = breast_model.predict(scaled_input)[0]
            result_label = "Malignant" if prediction == 1 else "Benign"

            BreastCheckup.objects.create(
                user=user,
                age=age,
                menopause=menopause,
                tumor_size=tumor_size,
                inv_nodes=inv_nodes,
                breast=breast,
                metastasis=metastasis,
                breast_quadrant=breast_quadrant,
                history=history,
                result=prediction
            )

            return render(request, "breast_cancer_results.html", {
                "form_data": form_data,
                "result": result_label
            })

        except Exception as e:
            messages.error(request, f"Invalid input: {e}")
            return render(request, "breast_cancer_checkup.html", {"form_data": form_data})

    return render(request, "breast_cancer_checkup.html", {
        "form_data": {}
    })

