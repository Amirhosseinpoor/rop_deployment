from django.contrib.auth.models import Group
from django.db import connection
from django.db.utils import ProgrammingError, OperationalError
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.contrib.auth import get_user_model
from .models import UserProfile

User = get_user_model()

def _table_exists(name: str) -> bool:
    try:
        return name in connection.introspection.table_names()
    except Exception:
        return False

def ensure_group(name):
    grp, _ = Group.objects.get_or_create(name=name)
    return grp

@receiver(post_save, sender=User)
def create_profile_for_user(sender, instance, created, **kwargs):
    """
    Safely ensure a profile exists after user creation.
    """
    if not _table_exists('usac_userprofile'):
        return
    if created:
        try:
            UserProfile.objects.get_or_create(user=instance, defaults={'role': UserProfile.ROLE_EMPLOYEE})
        except (ProgrammingError, OperationalError):
            return

@receiver(post_save, sender=UserProfile)
def sync_group_membership(sender, instance, **kwargs):
    """
    Keep Django auth groups in sync with the profile role.
    """
    try:
        mg = ensure_group('manager')
        dc = ensure_group('doctor')
        em = ensure_group('employee')

        instance.user.groups.clear()
        role_map = {'manager': mg, 'doctor': dc, 'employee': em}
        if instance.role in role_map:
            instance.user.groups.add(role_map[instance.role])
    except (ProgrammingError, OperationalError, KeyError):
        return
