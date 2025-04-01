import pandas as pd
import random

# Define categories and sample texts
categories = {
    "legal": [
        "The court granted the motion to dismiss based on lack of jurisdiction.",
        "The defendant was found guilty of fraud and sentenced to five years in prison.",
        "The contract was voided due to a breach of terms.",
        "The judge issued an injunction to prevent further violations.",
        "A new law was passed to regulate online privacy.",
        "The plaintiff filed a lawsuit against the company for negligence.",
        "Intellectual property laws protect creators from unauthorized use of their work.",
        "The attorney argued that the evidence was inadmissible in court."
    ],
    "business": [
        "Quarterly earnings exceeded analyst expectations by 10%.",
        "The company announced a merger with its largest competitor.",
        "Stock prices surged after the acquisition deal was confirmed.",
        "The startup secured $50 million in Series B funding.",
        "Market trends indicate a rise in consumer spending.",
        "A new marketing strategy increased revenue by 20%.",
        "The business expanded into international markets for growth.",
        "Corporate social responsibility initiatives improved brand reputation."
    ],
    "medical": [
        "The patient's lab results showed elevated white blood cell count.",
        "Doctors recommend annual check-ups for early disease detection.",
        "A new vaccine was developed to combat the virus outbreak.",
        "Research suggests a link between diet and heart disease.",
        "The hospital implemented new safety protocols for surgery.",
        "The surgeon performed a groundbreaking transplant operation.",
        "Mental health awareness campaigns have increased in recent years.",
        "A study found that exercise reduces the risk of chronic illness."
    ],
    "sports": [
        "The quarterback threw for 300 yards and 3 touchdowns in the game.",
        "The soccer team secured a last-minute victory in the finals.",
        "The athlete broke the world record for the 100m sprint.",
        "The championship match ended in a dramatic penalty shootout.",
        "A new coach was appointed to lead the basketball team.",
        "The Olympic games showcased incredible athletic performances.",
        "A tennis player won the Grand Slam after an intense final.",
        "The team's defensive strategy helped secure the championship title."
    ],
    "entertainment": [
        "The new film sequel broke box office records during its opening weekend.",
        "The singer released a new album that topped the charts.",
        "A popular TV series announced its final season.",
        "The award ceremony celebrated the best performances of the year.",
        "A well-known director revealed plans for a new sci-fi movie.",
        "A famous actor starred in a critically acclaimed drama.",
        "Streaming platforms are changing the way people consume media.",
        "Music festivals attract thousands of fans from around the world."
    ]
}

# Generate dataset with balanced categories
data = []
for category, texts in categories.items():
    for _ in range(1000):  # Increase number of samples per category
        text = random.choice(texts)
        data.append([text, category])

# Create DataFrame and save as CSV
df = pd.DataFrame(data, columns=["text", "category"])
df.to_csv("text_classification_dataset.csv", index=False)

print("Dataset generated successfully with", len(df), "samples.")
