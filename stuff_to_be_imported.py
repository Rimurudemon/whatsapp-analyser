high_examples = [

    # Hinglish : English (High Similarity)
    ("Mujhe pani chahiye", "I need water"),
    ("Aaj mausam accha hai", "The weather is nice today"),
    ("Tum kahan ja rahe ho?", "Where are you going?"),
    ("Mujhe naya phone lena hai", "I want to buy a new phone"),
    ("Main thoda busy hoon", "I am a bit busy"),

    # Hinglish : Hinglish (High Similarity)
    ("Kal holiday hai", "Kal chhutti hai"),  # Synonyms
    ("Tum kya kar rahe ho?", "Tu kya kar raha hai?"),  # Casual same intent
    ("Mujhe neend aa rahi hai", "Mujhe sona hai"),  # Related sleep phrases
    ("Yeh kitna mehenga hai!", "Kitni costly cheez hai!"),  # Same meaning
    ("Maine pizza order kiya", "Maine burger order kiya"),  # Same act, slight variation

    # English : English (High Similarity)
    ("I am going to sleep", "I am heading to bed"),  # Near perfect
    ("She likes chocolates", "She loves sweets"),  # Similar preferences
    ("We are watching TV", "We are watching a show"),  # Same activity
    ("The dog is barking", "The dog is making noise"),  # Same event
    ("I am learning coding", "I am studying programming"),  # Almost identical

    # Cross Mix (High Similarity)
    ("Main tired hoon", "I am feeling tired"),  # Near translation
    ("Usne mujhe gift diya", "She gave me a gift"),  # Near translation
    ("Kal result aayega", "Tomorrow the result will come"),  # Translation
    ("Yeh mera favorite color hai", "This is my favorite color"),  # Translation
    ("Woh bohot smart hai", "He is very intelligent"),  # Close meaning
]


moderate_examples = [

    # Hinglish : English (Moderate)
    ("Main ghar jaa raha hoon", "I am going to the office"),  # Same structure, diff place
    ("Mujhe chai pasand hai", "I like coffee"),  # Similar preference, diff object
    ("Aaj exam hai", "Today is the holiday"),  # Opposite intent
    ("Usne mujhe call kiya", "I messaged him"),  # Similar communication
    ("Bahut badiya movie thi", "The food was delicious"),  # Positive sentiment, diff domains

    # Hinglish : Hinglish (Moderate)
    ("Woh kal hospital gaya tha", "Main kal mall gaya tha"),  # Similar pattern, different place
    ("Kal party thi", "Kal boring lecture tha"),  # Event vs boring event
    ("Maine nayi car kharidi", "Maine naya laptop kharida"),  # Purchase, diff item
    ("Yeh kitni sundar hai", "Yeh kitna acha hai"),  # Adjective, diff object
    ("Mujhe doston ke sath time spend karna hai", "Mujhe akela rehna hai"),  # Contrasting preference

    # English : English (Moderate)
    ("I am feeling cold", "The room is warm"),  # Related but contrasting
    ("She loves dogs", "He likes cats"),  # Similar verb, different object
    ("We are going shopping", "They are going for a movie"),  # Outing events, diff nature
    ("He is writing a book", "She is reading a book"),  # Writing vs reading
    ("I will visit Paris", "I want to go to London"),  # Travel, diff places

    # Cross Mix (Moderate)
    ("Main movie dekhne jaa raha hoon", "I am going shopping"),  # Similar activity structure, diff activity
    ("Woh abhi ghar par hai", "She is outside"),  # Opposites
    ("Mujhe garmi lag rahi hai", "It is cold today"),  # Opposite weather
    ("Main lunch kar raha hoon", "I am working now"),  # Different actions
    ("Kal result hai", "I am expecting an email"),  # Similar expectation feel
]


low_examples = [

    # Hinglish : English (Low)
    ("Main pizza khana chahta hoon", "The stock market crashed"),  # Unrelated
    ("Aaj mausam accha hai", "I lost my keys"),  # Different domains
    ("Kal birthday hai", "The traffic is terrible"),  # Unrelated context
    ("Mujhe chai pasand hai", "The computer is not working"),  # Different topics
    ("Woh mujhe pasand hai", "It is raining outside"),  # Random relation

    # Hinglish : Hinglish (Low)
    ("Mujhe gym jana hai", "Aaj bazaar band hai"),  # Different events
    ("Kal cricket match hai", "Mujhe painting karni hai"),  # Hobby vs sports
    ("Maine usse message kiya", "Mera fridge kharab ho gaya"),  # Communication vs object
    ("Tumhare kapde ache hai", "Mujhe maths padhni hai"),  # Clothes vs study
    ("Main train pakad raha hoon", "Mere shoes gande hai"),  # No relation

    # English : English (Low)
    ("The sun is shining", "I forgot my wallet"),  # Unrelated
    ("He loves programming", "The food tastes awful"),  # Tech vs food
    ("They are dancing", "The printer is broken"),  # No connection
    ("The child is crying", "I will travel tomorrow"),  # Random
    ("We are at the beach", "My laptop is slow"),  # Different settings

    # Cross Mix (Low)
    ("Mujhe pani chahiye", "The sky is blue"),  # Needs vs description
    ("Main abhi office mein hoon", "I love playing guitar"),  # Place vs hobby
    ("Woh cricket khelta hai", "She is cooking dinner"),  # Activity, diff context
    ("Maine phone charge kiya", "It is snowing outside"),  # Gadget vs weather
    ("Usne mujhe ignore kiya", "The train is late"),  # Emotion vs event
]

english_stopwords = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
    'theirs', 'themselves', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'would', 'should', 'could', 'ought', 'can', 'will', 
    'just', 'don', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'wasn', 'weren', 'won', 'wouldn', 'shan', 'shouldn',
    'mustn', 'mightn', 'needn', 'aren', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 
    'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 
    'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'now', 'also', 'even', 'though', 'yet', 'still'
}

hinglish_stopwords = {
    'hai', 'nahi', 'kya', 'ka', 'ki', 'ke', 'ho', 'tha', 'the', 'thi', 'to', 'se', 'mein', 'par', 'bhi', 'ye', 'vo',
    'tum', 'main', 'mera', 'tera', 'unki', 'unka', 'ham', 'hum', 'ab', 'kal', 'kyun', 'kaise', 'zaroori', 'acha', 
    'accha', 'sab', 'kuch', 'kitna', 'bahut', 'mat', 'lekin', 'hona', 'hota', 'hoti', 'hote', 'lo', 'le', 'kar', 
    'karo', 'karta', 'karti', 'karte', 'tha', 'thi', 'the', 'hoon', 'raha', 'rahi', 'rahe', 'tha', 'thi', 'the', 
    'gaya', 'gayi', 'gaye', 'liye', 'wala', 'wali', 'wale', 'aur', 'ya', 'isse', 'uske', 'iske', 'unke', 'inka', 
    'unka', 'unki', 'unka', 'isne', 'usne', 'isko', 'usko', 'tak', 'jab', 'tab', 'abhi', 'fir', 'phir', 'pe', 'pehle', 
    'baad', 'kahi', 'kisi', 'kisi', 'sabhi', 'har', 'jo', 'jis', 'jise', 'jiska', 'jiske', 'jiski', 'yaha', 'waha', 
    'yahan', 'wahan', 'kitne', 'kitni', 'kyunki', 'kyonki', 'humesha', 'kab', 'kahan', 'kaun', 'kiska', 'kisne', 
    'kiski', 'kis', 'kuchh', 'acha', 'acchi', 'accha', 'acche', 'bohot', 'bahut', 'thoda', 'zyada', 'kam', 'lagbhag',
    'sabse'
}
