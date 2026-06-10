from celery import shared_task
from django.core.mail import EmailMultiAlternatives
from django.template.loader import render_to_string
@shared_task
def email_sending(email, username):

    subject = 'Welcome to Mediverse AI!'
    from_email = None
    to = [email]

    # Render HTML Email
    html_content = render_to_string('usac/emails/welcome_email.html', {'username': username})
    text_content = f"Hi {username}, welcome to Mediverse AI!"  # نسخه متنی ساده

    msg = EmailMultiAlternatives(subject, text_content, from_email, to)
    msg.attach_alternative(html_content, "text/html")
    msg.send()