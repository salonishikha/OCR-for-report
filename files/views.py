# Create your views here.
from django.views.generic.edit import FormView
from django.http.response import HttpResponse
from files.forms import ProfileImageForm
from files.models import ProfileImage


class ProfileImageView(FormView):
    # template_name = 'profile_image_form.html'
    template_name = 'front.html'
    form_class = ProfileImageForm

    def form_valid(self, form):
        profile_image = ProfileImage(
            image=self.get_form_kwargs().get('files')['image'],
            excelfile= self.get_form_kwargs().get('files')['excelfile'],
        )
        profile_image.save()

        # self.id = profile_image.id
        return HttpResponse(' Uploaded')
