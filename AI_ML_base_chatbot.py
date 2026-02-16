# WordPress Chatbot with Question Variations
import pandas as pd
from rapidfuzz import process
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time

# =========================
# Mapping question variations to single response
# =========================
qa_mapping = {
    # Basic greetings
    ("hi","hello","hey","hey there","hi there","good morning","morning","good afternoon","afternoon","good evening","evening","howdy","hiya","greetings","what's up","sup","yo","how are you","how's it going","how do you do"): "Hello! How can I help you?",

    ("how are you","how's it going","how do you do","how are you doing","how have you been","how's everything","how's your day","what's up","how's life","are you fine"): "I am fine, thank you!",
    
    ("what is your name","who are you","tell me your name","your name please","introduce yourself","who am I talking to","can you tell me your name","what should I call you","what do I call you","name"): "My name is ChatPy. I am a WordPress chatbot.",

    ("bye","goodbye","see you","see you later","talk to you later","catch you later","farewell","bye bye","have a good day","later"): "Goodbye! Have a great day 😊",

    ("good morning","morning","morning greetings","morning hi","morning hello","good day","hi good morning","hello good morning","morning everyone","morning chatpy"): "Good morning! Hope you have a nice day ☀️",

    ( "good night","night","night greetings","night hi","night hello","sleep well","have a good night","night chatpy","good night chatpy","hello good night"): "Good night! Sleep well 🌙",
    
    ( "thank you","thanks","thx","thank you very much","thanks a lot","thanks chatpy","thank you chatpy","much appreciated","thank you so much","thanks a bunch"): "You are welcome!",

    # About WordPress
    ("what is wordpress","define wordpress","wordpress meaning","explain wordpress","wordpress is","tell me about wordpress","wordpress info","wordpress overview","what does wordpress do","wordpress cms"):"WordPress is a free tool to make websites.You can use it without knowing how to code.It has ready-made designs called themes and extra tools called plugins.You can make blogs, business sites, or online stores.Example: A bakery can make a website with a menu and order page using WordPress.",
    
    ("how to download wordpress","where to download wordpress","wordpress download","get wordpress","download wordpress","wordpress download link","where can I download wordpress","how do I download wordpress","how i start wordpress","wordpress official download","download wordpress from website"):"Go to wordpress.org.Click Get WordPress and download it.Open the downloaded file to start using WordPress.",

    ( "is wordpress easy to use","wordpress beginner friendly","do I need coding for wordpress","wordpress requires coding","wordpress simple to use","can a beginner use wordpress","wordpress no coding needed","wordpress user friendly","is wordpress difficult","wordpress easy for beginners"):"WordPress is easy to use and does not require coding skills.Example:Even if you don’t know HTML or CSS, you can make a blog, a business website, or an online store using WordPress by just choosing a theme and adding content.",

    ("what are wordpress features","wordpress features","wordpress functionality","wordpress options","wordpress capabilities","what can wordpress do","wordpress tools","wordpress plugins and themes","wordpress seo features","wordpress customization"):"WordPress features include themes, plugins, SEO, and customization.Explanation:Themes: Ready-made designs for your website.Plugins: Add extra features like contact forms or online stores.SEO: Helps your website appear in search engines.Customization: Change colors, fonts, and layouts easily.Example:A photographer can use a gallery theme, add a booking plugin, optimize images for search, and change the site’s look to match their style.",

    ( "why is wordpress important","importance of wordpress","why use wordpress","why choose wordpress","benefits of wordpress","wordpress advantages","why wordpress is useful","why learn wordpress","wordpress significance","wordpress value"):"WordPress is important because it helps create websites quickly.Explanation:With WordPress, you don’t need to code from scratch. You can use ready-made themes and plugins to build a website fast.Example:A small business can set up an online store in a few hours using WordPress instead of hiring a developer.",
        
    ("why learn wordpress","benefits of learning wordpress","learning wordpress advantages","why should I learn wordpress","how learning wordpress helps me","wordpress skills importance","earn online with wordpress","wordpress for making money","learning wordpress for websites","wordpress learning benefits"):"Explanation:If you know WordPress, you can create websites for yourself or for others. Many people also earn money by freelancing, selling themes, or managing websites.Example:A student can make websites for small businesses and earn money as a WordPress developer",
    
    ("where can I learn wordpress","how to learn wordpress","wordpress learning resources","best way to learn wordpress","learn wordpress online","wordpress tutorials","wordpress courses","wordpress guides","wordpress learning platforms","wordpress documentation"):"There are many free and paid resources to learn WordPress. You can watch video tutorials, take online courses, or read the official guides step by step.Example:A beginner can watch a YouTube tutorial to create a blog or follow Udemy courses to make a professional website.",
    
    ("which language does wordpress use","wordpress programming language","wordpress coding language","what languages wordpress is built with","wordpress uses php html css javascript","languages needed for wordpress","wordpress tech stack","wordpress coding requirements","wordpress development languages","wordpress backend languages"):"WordPress mainly uses PHP, HTML, CSS, and JavaScript.",
    
    ("what knowledge is required to start wordpress","wordpress prerequisites","do I need coding skills for wordpress","how much knowledge is needed for wordpress","wordpress beginner requirements","can I start wordpress without coding","what is needed to use wordpress","wordpress basics required","skills needed for wordpress","wordpress starter knowledge"):"Basic computer and internet knowledge is enough to start WordPress.",

    # Dashboard
    ("what is wordpress dashboard","define wordpress dashboard","what is dashboard in wordpress","wordpress dashboard meaning","dashboard overview wordpress","wordpress admin dashboard","wordpress dashboard features","control panel of wordpress","wordpress backend dashboard","what does dashboard do in wordpress"):"The WordPress dashboard is the control panel of the website.From the dashboard, you can add pages, posts, and media, install themes and plugins, and manage your website settings. It’s where you control everything on your site.",
    
    ("what can I do from wordpress dashboard","manage posts pages themes plugins in dashboard","how to use wordpress dashboard","dashboard functions in wordpress","wordpress dashboard options","what is included in wordpress dashboard","using dashboard to manage wordpress","dashboard features wordpress","how to control wordpress from dashboard","wordpress admin panel functions"):"You can manage posts, pages, themes, and plugins from the dashboard.",
    
    ("what is in wordpress dashboard","what does dashboard contain in wordpress","wordpress dashboard contents","dashboard items in wordpress","posts pages media plugins settings in dashboard","what features are in wordpress dashboard","dashboard overview wordpress","what sections are in wordpress dashboard","components of wordpress dashboard","what can I find in wordpress dashboard"):"Dashboard includes posts, pages, media, plugins, and settings.",
    
    ("what is home tab in wordpress dashboard","define home tab in wordpress","what does the home tab show","home tab overview wordpress","what is shown in home tab","dashboard home tab features","wordpress dashboard home tab","site overview in home tab","wordpress home tab updates","what information is in home tab"):"The Home tab shows site overview and updates.",
    
    ( "what are updates in wordpress","wordpress update meaning","define updates in wordpress","how to check updates in wordpress","wordpress new version updates","wordpress dashboard updates","themes and plugins updates wordpress","what does updates section show","update wordpress site","check for updates in wordpress"):"Updates show new versions of WordPress, themes, and plugins.",
    
    ("how to open wordpress dashboard","access wordpress dashboard","login to wordpress dashboard","wordpress admin login","how to reach wordpress dashboard","wordpress dashboard url","wp-admin access","open wp-admin page","where is wordpress dashboard","how do I get to wordpress dashboard"):"You can open the dashboard by adding /wp-admin to the website URL.",

    # Posts
    ("what are posts in wordpress","define posts in wordpress","purpose of posts in wordpress","wordpress posts meaning","how are posts used in wordpress","posts section in wordpress","what is post in wordpress","what do posts do in wordpress","wordpress blog posts","how to use posts in wordpress"):"Posts are used to publish blog content in WordPress.",
    
    ("what is all posts in wordpress","all posts list wordpress","where to see all posts","show all posts wordpress","wordpress list of posts","view all blog posts wordpress","all wordpress posts","list of published posts wordpress","how to see all posts wordpress","wordpress posts overview"):"All posts show the list of published posts.",
    
    ("what is add new post in wordpress","how to add new post","create new post wordpress","how to create a blog post","add post wordpress","wordpress new post","make a new post","how to publish a post in wordpress","add blog post wordpress","creating posts in wordpress"):"Add New Post is used to create a new blog post.",
    
    ("what are categories in wordpress","define categories in wordpress","wordpress categories meaning","how to use categories wordpress","categories for posts wordpress","organize posts with categories","wordpress post categories","what do categories do in wordpress","how to manage categories wordpress","categories section in wordpress"):"Categories help organize posts into topics.",
    
    ("what are tags in wordpress","define tags in wordpress","wordpress tags meaning","how to use tags wordpress","tags for posts wordpress","describe posts with tags","wordpress post tags","what do tags do in wordpress","how to manage tags wordpress","tags section in wordpress"):"Tags describe post details and keywords.",

    # Media
    ("what is media in wordpress","define media in wordpress","wordpress media meaning","how to use media wordpress","manage media files wordpress","media section in wordpress","wordpress images and videos","media library wordpress","upload media wordpress","what does media do in wordpress"):"Media is used to manage images, videos, and files.",
    
    ("what is media library in wordpress","define media library wordpress","wordpress media library meaning","how to use media library wordpress","media library section wordpress","where are uploaded files in wordpress","wordpress file storage","media library stores files","all uploaded files wordpress","manage media library wordpress"):"Media library stores all uploaded files.",
    
    ("how to add media files in wordpress","how to upload media wordpress","add new media wordpress","upload images wordpress","upload videos wordpress","upload files wordpress","add files to media library","wordpress add media","how to add images and videos wordpress","upload content wordpress"): "You can upload files using the Add New Media option.",

    # Pages
    ("what are pages in wordpress","define pages wordpress","wordpress pages meaning","how to use pages wordpress","what is the purpose of pages wordpress","create pages wordpress","about page wordpress","contact page wordpress","difference between posts and pages","static pages in wordpress"):"Pages are used for static content like About and Contact pages.",
    
    ("how to add a new page in wordpress","create new page wordpress","add page wordpress","wordpress add new page","how to create a page wordpress","new page wordpress","add about page wordpress","add contact page wordpress","add static page wordpress","wordpress create page"):"You can create a new page using the Add New Page option.",
    
    ("difference between posts and pages","posts vs pages","pages vs posts","how posts differ from pages","what is the difference between posts and pages","dynamic vs static content wordpress","are pages static or dynamic","are posts static or dynamic","how posts and pages are different","posts and pages difference"):"Posts are dynamic while pages are static.",

    # Plugins
    ("what are plugins in wordpress","what are plugins","what is plugins","define plugins wordpress","wordpress plugins meaning","purpose of plugins wordpress","how plugins work in wordpress","wordpress plugins explained","plugins in wordpress","what do plugins do wordpress","add plugins wordpress","install plugins wordpress"):"Plugins add extra features to WordPress.",
    
    ("how to install plugins in wordpress","install plugins wordpress","add plugins wordpress","wordpress install plugin","where to install plugins wordpress","plugins installation wordpress","how to add plugin wordpress","wordpress plugin setup","wordpress plugins section","how do i install plugins in wordpress"):"You can install plugins from the Plugins section.",
    
    ("best seo plugins wordpress","seo plugins wordpress","popular seo plugins","yoast seo plugin","rank math plugin","wordpress seo plugin","which seo plugins are best","top seo plugins wordpress","seo plugin recommendations","seo tools wordpress"):"Yoast SEO and Rank Math are popular SEO plugins.",
    
    ("wordpress form plugins","best form plugins wordpress","create forms wordpress","how to make forms wordpress","contact form 7 plugin","wpforms plugin","wordpress form builder","add forms wordpress","form plugins for wordpress","which plugins for forms wordpress"):"Contact Form 7 and WPForms are used to create forms.",
    
    ("woocommerce plugin","how to use woocommerce","wordpress online store plugin","create online store wordpress","best ecommerce plugin wordpress","wordpress ecommerce","woocommerce setup","how to make online store wordpress","add woocommerce wordpress","woocommerce tutorial"):"WooCommerce is used to create an online store.",

    # Settings
    ("wordpress settings","how to use wordpress settings","settings in wordpress","website settings wordpress","manage wordpress settings","wordpress site configuration","change wordpress settings","wordpress general settings","configure wordpress website","wordpress admin settings"):"Settings control the overall configuration of the website.",
    
    ("what are permalinks in wordpress","wordpress permalinks","permalinks settings wordpress","how to change permalinks wordpress","url structure wordpress","wordpress url settings","edit permalinks wordpress","wordpress link structure","custom permalinks wordpress","permalinks configuration wordpress"):"Permalinks control the URL structure of your website.",
    
    ("what is privacy in wordpress","wordpress privacy settings","manage privacy wordpress","wordpress data protection","privacy options wordpress","user privacy wordpress","wordpress privacy configuration","how to change privacy settings wordpress","wordpress gdpr settings","wordpress privacy policy settings"):"Privacy settings manage user data protection.",

    ("how to create a page in wordpress", "wordpress create new page", "add page wordpress", "wordpress page editor", "create static page wordpress", "wordpress page setup", "wordpress add about page", "wordpress page management", "wordpress page tutorial", "wordpress create homepage page") : "Go to the WordPress dashboard.Click Pages → Add New.Enter a title and content for your page.Click Publish to make it live.",

    ("how to create a post in wordpress", "wordpress add blog post", "wordpress new post tutorial", "wordpress post editor", "publish post wordpress", "wordpress write article", "wordpress post management", "wordpress add news post", "wordpress blog post setup", "wordpress create post guide") :"Go to the WordPress dashboard.Click Posts → Add New.Enter a title and write your content.Add categories or tags if needed.Click Publish to post it on your website.",

    ("how to create a website in wordpress", "wordpress website tutorial", "wordpress setup website", "wordpress build site", "wordpress website creation guide", "wordpress start new site", "wordpress make website", "wordpress beginner website tutorial", "wordpress full site setup", "wordpress website building steps") :"Download and install WordPress from wordpress.org or use a hosting service with WordPress pre-installed.Log in to the WordPress dashboard.Choose a theme under Appearance → Themes.Add pages and posts as needed.Customize your website using themes, plugins, and settings.Click Publish to make your website live."
}
# =========================
# Expand dictionary into lists for DataFrame
# =========================
questions = []
responses = []

for keys, response in qa_mapping.items():
    for q in keys:
        questions.append(q)
        responses.append(response)

# =========================
# Initialize Sentence Transformer for embeddings
# =========================
model_embed = SentenceTransformer('all-MiniLM-L6-v2')
question_vectors = model_embed.encode(questions)

# =========================
# Function to get best response
# =========================
def get_response(user_input):
    # Encode input
    user_vec = model_embed.encode([user_input])
    
    # Compute cosine similarity
    similarity = cosine_similarity(user_vec, question_vectors)
    best_index = np.argmax(similarity)
    best_score = similarity[0][best_index]

    # Threshold for unknown questions
    if best_score < 0.5:
        return "Sorry, I don’t understand. Can you rephrase?"
    return responses[best_index]

# =========================
# Chat Function
# =========================
def chatbot():
    print("AI Chatbot 🤖: Hello! Type 'bye' to exit.")
    while True:
        user_input = input("You: ").lower()
        if user_input == "bye":
            print("AI Chatbot 🤖: Goodbye! Have a nice day 😊")
            break

        response = get_response(user_input)

        # Typing effect for nicer UX
        for char in response:
            print(char, end='', flush=True)
            time.sleep(0.01)
        print("\n")

# =========================
# Run Chatbot
# =========================
if __name__ == "__main__":
    chatbot()