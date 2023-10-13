from http.server import BaseHTTPRequestHandler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Sample provider descriptions as single paragraphs
provider_descriptions = [
    "I am an experienced lawyer specializing in family law. With over 15 years of practice, I've successfully handled numerous divorce and child custody cases, providing comprehensive legal support to my clients. My dedication to upholding the rights of families sets me apart.",
    "As an expert mediator, I specialize in resolving civil disputes. My extensive experience in arbitration and mediation spans a decade, helping businesses and individuals find amicable solutions to complex conflicts. I've successfully mediated over 100 cases.",
    "I am a notary public with a strong focus on document notarization services. My commitment to accuracy and integrity ensures that all documents I notarize meet legal requirements. I've notarized documents for over 500 satisfied clients.",
    "With a background in corporate law, I've assisted numerous businesses in navigating complex legal landscapes. I've successfully represented companies in various industries, providing legal solutions that drive growth and protect their interests.",
    "I am an immigration lawyer, specializing in visa applications and immigration issues. My track record includes helping over 200 clients secure their visas and start new chapters in their lives in a new country.",
    "With a passion for criminal defense, I've defended countless clients in courtrooms, securing their rights and ensuring justice is served. My legal expertise has led to the exoneration of multiple wrongfully accused individuals.",
    "As a personal injury attorney, I've fought for justice on behalf of injured individuals. I've won millions in settlements for my clients, holding those responsible accountable for their negligence.",
    "I am a real estate lawyer with a focus on property transactions. I've facilitated the smooth transfer of properties for over 300 clients, ensuring their investments are protected and transactions are legally sound.",
    "Specializing in intellectual property law, I've helped numerous startups and businesses protect their innovations through patent and trademark registrations. My work has contributed to the success of many startups.",
    "I am a family mediator with a background in psychology. I use my expertise to facilitate healthy family discussions and resolutions. I've helped families navigate challenging situations and find common ground.",
    "With a passion for human rights law, I've dedicated my career to advocating for marginalized communities in India. My budget-conscious legal support ensures access to justice for those in need. My legal education includes a degree from a prominent Indian law school.",
    "I'm a personal injury attorney in India, specializing in cases of medical negligence. My budget-friendly legal representation has secured compensation for numerous injured individuals. I hold a postgraduate degree in medical law and ethics",
    "I'm a seasoned Indian lawyer with over two decades of experience in civil and criminal law. I've successfully represented clients in high-profile cases while offering affordable legal services. Holding a master's degree in law from a prestigious Indian university, I'm committed to upholding justice",
    "With a focus on intellectual property law in India, I've assisted tech startups and innovators in protecting their inventions. My budget-conscious approach has facilitated innovation while holding a master's degree in intellectual property law",
    "I'm an immigration lawyer in India with a focus on visa applications and citizenship. My budget-conscious services have helped individuals find a new home in India. I hold a master's degree in immigration law from an Indian university.",
    "I am an Indian corporate attorney, specializing in mergers and acquisitions. My budget-friendly legal services have enabled small businesses to achieve growth and success. I possess an MBA in addition to a law degree.",
    "With a background in taxation law, I've assisted Indian startups and entrepreneurs in navigating complex tax regulations. My budget-conscious legal counsel has contributed to their financial success. I possess a chartered accountant qualification in addition to a law degree",
    "I'm an Indian notary public with a strong emphasis on property-related document notarization. My attention to detail ensures the legality of documents at affordable rates. I also hold a diploma in notarial practice from an Indian legal academy."
]

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(provider_descriptions)

def print_recommended_providers(user_query):
    # Transform the user query
    query_tfidf = tfidf_vectorizer.transform([user_query])

    # Compute cosine similarity between the query and provider descriptions
    cosine_similarities = linear_kernel(query_tfidf, tfidf_matrix)

    # Get recommendations based on the highest cosine similarity
    recommendations = cosine_similarities[0].argsort()[:-4:-1]  # Get the top 10 recommendations

    print("Recommended Providers:")
    for rank, provider in enumerate(recommendations, start=1):
        print(f"Rank {rank}: {provider_descriptions[provider]}")

user_query = "I'm a real estate developer and require legal support for property transactions, zoning issues, and land-use regulations. I need a real estate attorney who understands the complexities of the real estate industry."
print_recommended_providers(user_query)
 
class handler(BaseHTTPRequestHandler):
 
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type','text/plain')
        self.end_headers()
        self.wfile.write('Hello, world!'.encode('utf-8'))
        return