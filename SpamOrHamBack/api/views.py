from .serializers import EmailSerializer
from .models import classify_spam
from rest_framework.views import APIView
from rest_framework.response import Response

class SpamClassificationView(APIView):
    def post(self, request):
        serializer = EmailSerializer(data=request.data)
        if serializer.is_valid():
            email = serializer.validated_data['email']
            result = classify_spam(email)

            return Response({'result': result})
        else:
            return Response(serializer.errors, status=400)