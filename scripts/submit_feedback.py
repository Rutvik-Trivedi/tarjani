import requests

url = "https://tarjani-github.000webhostapp.com/"
print("***Thank you for your interest in TARJANI. Your feedback shall help me improve TARJANI even better***")
print("You can also submit your feedback and suggestions by visiting http://tarjani.is-great.net")
print("Please enter the requested information for submitting your feedback")

print()
name = input("Please enter your name (This field is optional. Leave empty to skip this): ")
email = input("If you want me to contact you back, please provide your email ID. Leave it empty to skip: ")
subject = input("Please provide a subject line for your feedback (This field is required. Minimum 3 characters required): ")
message = input("Please provide your message to submit (This field is required. Minimum 3 characters required): ")
print()
print("Submitting your feedback")

data = {
"name": name,
"email": email,
"subject": subject,
"message": message}
try:
    r = requests.post(url = url, data = data)
    print(r.text)
except requests.exceptions.ConnectionError:
    print("Unable to connect to TARJANI's Feedback Notebook. Please check your connection")
