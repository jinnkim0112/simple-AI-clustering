# from dotenv import load_dotenv
# load_dotenv() 
import base64
from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage, AIMessage

def encode_image(uploadedImage):
    return base64.b64encode(uploadedImage.read()).decode('utf-8')

def clustering(uploadedFileImage, classes):
    chain1 = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=1024)
    image = encode_image(uploadedFileImage)

    msg1 = chain1.invoke(
        [   AIMessage(
                content="You are a useful bot that is especially good at summarizing images"
            ),
            HumanMessage(
                content=[
                    {"type": "text", "text": "Characterize the image using a well-detailed description"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpg;base64,{image}"
                        },
                    },
                ]
            )
        ]
    )

    chain2 = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=1024)
    humanMessageOrder = (
        'I will give you an image and a text. The text is the descriptin of the image.\n'
        'Please identify as one of the following classes: \n'
        "" + ','.join(classes) + '\n'
        'Only return the name of the class. No additional explanation or anything related. Just the name.\n'
        'Example:\n'
        "'" + "'\n'".join(classes) + "'" 
    ) 

    msg2 = chain2.invoke(
        [   AIMessage(
                content="You are a useful bot that is especially good at classifing images when the image and text is given"
            ),
            HumanMessage(
                content=[
                    {"type": "text", "text": humanMessageOrder},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpg;base64,{image}"
                        },
                    },
                    {"type": "text", "text": msg1.content}
                ]
            )
        ]
    )

    return msg2.content