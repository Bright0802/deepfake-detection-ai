import torch
import torch.nn as nn
import cv2
from torchvision import transforms
from PIL import Image

class DeepfakeCNN(nn.Module):
    def __init__(self):
        super(DeepfakeCNN,self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3,16,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16,32,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(32*32*32,128),
            nn.ReLU(),
            nn.Linear(128,2)
        )

    def forward(self,x):
        x = self.conv(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x


model = DeepfakeCNN()
model.load_state_dict(torch.load("deepfake_model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

cap = cv2.VideoCapture(0)

while True:

    ret,frame = cap.read()

    img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = transform(img).unsqueeze(0)

    output = model(img)
    _,pred = torch.max(output,1)

    label = "Real" if pred.item()==0 else "Deepfake"

    cv2.putText(frame,label,(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow("Deepfake Detector",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()