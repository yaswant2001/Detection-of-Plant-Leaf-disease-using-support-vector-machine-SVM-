import cohere

co = cohere.ClientV2("u5pQVCcMHfo1elOzDpshVmUKy1eJ7KDG79BeSLDK")

def get_remedies_from_google(disease):
    prompt = f"Suggest remedies for {disease} in rice leaf disease."
    
    response = co.chat(
        model="command-r-plus", 
        messages=[{"role": "user", "content": prompt}]
    )
    
    remedies_text = response.message.content
    formatted_remedies = format_remedies(remedies_text)
    return formatted_remedies


def format_remedies(remedy_items):
    formatted_text = []
    
    for remedy_item in remedy_items:
        remedy_text = remedy_item.text
        
        sections = remedy_text.split("\n\n")
        
        for section in sections:
            lines = section.split("\n")
            
            heading = lines[0]
            
            content = "\n".join([f"    {line}" for line in lines[1:]])
            
            formatted_text.append(f"{heading}:\n{content}")
    
    return "\n\n".join(formatted_text)
