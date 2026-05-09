import os
import json
import requests
from datetime import datetime
from PIL import Image
from io import BytesIO
from config import openai, weather_api_key,IMAGE_MODEL
# Add these:
import wikipedia
from tavily import TavilyClient
import xml.etree.ElementTree as ET
from requests.sessions import Session   # or just use requests

# Global session for arxiv
session = Session()


def wikipedia_search_tool(query: str, sentences: int = 5) -> list[dict]:
    """
    Searches Wikipedia for a summary of the given query.

    Args:
        query (str): Search query for Wikipedia.
        sentences (int): Number of sentences to include in the summary.

    Returns:
        list[dict]: A list with a single dictionary containing title, summary, and URL.
    """
    try:
        page_title = wikipedia.search(query)[0]
        page = wikipedia.page(page_title)
        summary = wikipedia.summary(page_title, sentences=sentences)

        return [{
            "title": page.title,
            "summary": summary,
            "url": page.url
        }]
    except Exception as e:
        return [{"error": str(e)}]
# --- artist ---
# Generates a weather-accurate pop-art city image using DALL-E.
# Called by handle_tool_call when the model decides an image is needed.
# The weather string (e.g. "Scattered clouds with 15.98°C") is injected into
# the prompt so sky colour, lighting, and atmosphere match real conditions.
def artist(city, weather="clear sky"):
    print(f"🎨 artist called for {city} with weather: {weather}")
    try:
        image_response = openai.images.generate(
            model=IMAGE_MODEL,
            prompt=(
                f"A vibrant pop-art style image of {city}. "
                f"Current weather: {weather}. "
                f"The scene must visually reflect these exact conditions: "
                f"sky colour, lighting, clothing on people, and atmosphere should all match the weather. "
                f"Show recognisable landmarks of {city}."
            ),
            size="1024x1024",
            n=1
        )

        image_url = image_response.data[0].url

        # Download and save the image locally
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))

        os.makedirs("images", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{city.lower().replace(',','').replace(' ','')}_{timestamp}.png"
        filepath = os.path.join("images", filename)
        img.save(filepath)

        print(f"✅ Image saved as {filepath}")

        try:
            img.show()
            print("👀 Image opened in default viewer.")
        except Exception as e:
            print(f"⚠️ Could not auto-open image: {e}")

        return filepath

    except Exception as e:
        print(f"❌ Image generation failed: {e}")
        return None





# Tool definition
wikipedia_tool_def = {
        "name": "wikipedia_search_tool",
        "description": "Searches for a Wikipedia article summary by query string.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords for the Wikipedia article."
                },
                "sentences": {
                    "type": "integer",
                    "description": "Number of sentences in the summary.",
                    "default": 5
                }
            },
            "required": ["query"]
        }

}

# --- get_weather ---
# Fetches live weather from OpenWeatherMap and returns a short description string
# e.g. "Scattered clouds with 15.98°C".
# This string is returned to the model as a tool result and can be passed to
# artist as-is so the image prompt reflects real conditions.
def get_weather(destination_city: str):
    print(f"🔧 get_weather called for {destination_city}")

    url = "http://api.openweathermap.org/data/2.5/weather"
    params = {"q": destination_city, "appid": weather_api_key, "units": "metric"}

    try:
        response = requests.get(url, params=params)
        print(f"🌍 Weather API status: {response.status_code}")
        print(f"🌍 Weather API raw: {response.text[:200]}")

        if response.status_code == 200:
            data = response.json()
            desc = data["weather"][0]["description"].capitalize()
            temp = data["main"]["temp"]
            # Format: "<Description> with <temp>°C" — reused verbatim in the artist prompt
            weather_text = f"{desc} with {temp}°C"
            return weather_text
        else:
            return f"Weather data not available for {destination_city}"
    except Exception as e:
        return f"Error fetching weather: {e}"


# --- Tool schemas ---
# These descriptions are the only guidance the model receives.
# No system message is used — the model decides when and how to call tools
# based solely on these descriptions.

weather_function_def = {
    "name": "get_weather",
    "description": "Get the current real-time weather for a city.",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {"type": "string"},
        },
        "required": ["destination_city"],
    },
}

artist_function_def = {
    "name": "artist",
    # Explicit instruction to only call when the user asks for an image,
    # preventing the model from always generating one after get_weather.
    "description": "Generate a pop-art style image of a city reflecting its current weather. Only call this when the user explicitly asks for an image.",
    "parameters": {
        "type": "object",
        "properties": {
            "city":    {"type": "string", "description": "City name"},
            "weather": {"type": "string", "description": "Weather description returned by get_weather. Defaults to 'clear sky' if not available."},
        },
        "required": ["city"],
    },
}



def tavily_search_tool(query: str, max_results: int = 5, include_images: bool = False) -> list[dict]:
    """
    Perform a search using the Tavily API.

    Args:
        query (str): The search query.
        max_results (int): Number of results to return (default 5).
        include_images (bool): Whether to include image results.

    Returns:
        list[dict]: A list of dictionaries with keys like 'title', 'content', and 'url'.
    """
    params = {}
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY not found in environment variables.")
    params['api_key'] = api_key

    #client = TavilyClient(api_key)

    api_base_url = os.getenv("DLAI_TAVILY_BASE_URL")
    if api_base_url:
        params['api_base_url'] = api_base_url

    client = TavilyClient(api_key=api_key, api_base_url=api_base_url)

    try:
        response = client.search(
            query=query,
            max_results=max_results,
            include_images=include_images
        )

        results = []
        for r in response.get("results", []):
            results.append({
                "title": r.get("title", ""),
                "content": r.get("content", ""),
                "url": r.get("url", "")
            })

        if include_images:
            for img_url in response.get("images", []):
                results.append({"image_url": img_url})

        return results

    except Exception as e:
        return [{"error": str(e)}]  # For LLM-friendly agents



tavily_tool_def = {
        "name": "tavily_search_tool",
        "description": "Performs a general-purpose web search using the Tavily API.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords for retrieving information from the web."
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return.",
                    "default": 5
                },
                "include_images": {
                    "type": "boolean",
                    "description": "Whether to include image results.",
                    "default": False
                }
            },
            "required": ["query"]
        }

}

def arxiv_search_tool(query: str, max_results: int = 5) -> list[dict]:
    """
    Searches arXiv for research papers matching the given query.
    """
    url = f"https://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}"

    try:
        response = session.get(url, timeout=60)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return [{"error": str(e)}]

    try:
        root = ET.fromstring(response.content)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}

        results = []
        for entry in root.findall('atom:entry', ns):
            title = entry.find('atom:title', ns).text.strip()
            authors = [author.find('atom:name', ns).text for author in entry.findall('atom:author', ns)]
            published = entry.find('atom:published', ns).text[:10]
            url_abstract = entry.find('atom:id', ns).text
            summary = entry.find('atom:summary', ns).text.strip()

            link_pdf = None
            for link in entry.findall('atom:link', ns):
                if link.attrib.get('title') == 'pdf':
                    link_pdf = link.attrib.get('href')
                    break

            results.append({
                "title": title,
                "authors": authors,
                "published": published,
                "url": url_abstract,
                "summary": summary,
                "link_pdf": link_pdf
            })

        return results
    except Exception as e:
        return [{"error": f"Parsing failed: {str(e)}"}]


arxiv_tool_def = {
    "name": "arxiv_search_tool",
        "description": "Searches for research papers on arXiv by query string.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords for research papers."
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return.",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    }








# --- Tool handler ---
# Executes whichever tool the model chose and returns a "tool" role message
# containing the result, which is appended to the conversation so the model
# can continue reasoning in the next loop iteration.
# Only message.tool_calls[0] is processed — one call per loop turn.
def handle_tool_call(name, args):
    """Dispatch a tool call by name and return the raw result."""
    if name == "get_weather":
        return get_weather(args.get("destination_city"))
    elif name == "artist":
        return artist(args.get("city"), args.get("weather", "clear sky"))
    elif name == "wikipedia_search_tool":
        return wikipedia_search_tool(**args)
    elif name == "tavily_search_tool":
        return tavily_search_tool(**args)
    elif name == "arxiv_search_tool":
        return arxiv_search_tool(**args)
    else:
        return {"error": f"Unknown tool: {name}"}


# ====================== TOOL WRAPPING ======================

weather_tool = {
    "type": "function",
    "function": weather_function_def
}

artist_tool = {
    "type": "function",
    "function": artist_function_def
}

wikipedia_tool = {
    "type": "function",
    "function": wikipedia_tool_def
}

tavily_tool = {
    "type": "function",
    "function": tavily_tool_def
}

arxiv_tool = {
    "type": "function",
    "function": arxiv_tool_def
}


# ====================== AVAILABLE TOOLS (Export this!) ======================
available_tools = [
    weather_tool,
    artist_tool,
    wikipedia_tool,
    tavily_tool,
    arxiv_tool,
]




if __name__ == "__main__":
    print("✅ Tools loaded successfully!")
    for i, tool in enumerate(available_tools):
        try:
            func = tool.get("function", {})
            name = func.get("name")
            print(f"   {i + 1:2}. {name if name else 'MISSING NAME!'}")
            if not name:
                print(f"      → Problematic tool: {tool}")
        except Exception as e:
            print(f"   {i + 1:2}. ERROR: {e}")
