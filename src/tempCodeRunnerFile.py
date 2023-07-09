#home layout
# Define the background image URL and center text
text = 'Lithofacies Identification'
background_image ="https://images.pexels.com/photos/162568/oil-pump-jack-sunset-clouds-silhouette-162568.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1"

# Define the layout of the app with the background image, centered text, and a paragraph
home_layout = html.Div(
    style={
        'background-image': f'url("{background_image}")',  # Set the background image URL
        'background-size': 'cover',  # Make sure the image covers the entire background
        'background-position': 'center',  # Center the background image
        'height': '100vh',  # Set the height of the container to the full height of the viewport
        'display': 'flex',  # Use a flexbox layout
        'flex-direction': 'column',  # Stack the elements vertically
        'align-items': 'center',  # Center the child elements horizontally
        'justify-content': 'center',  # Center the child elements vertically
        'font-family': 'Helvetica, Arial, sans-serif'  # Set the font family for the text
    },
    children=[
        html.H1(
            children=text,  # Use the text variable for the header text
            style={
                'font-size': '4.5em',  # Set the font size for the header
                "font-weight": "bold",  # Make the text bold
                'color': 'white',  # Set the text color to white
                'text-shadow': '2px 2px 4px #000000'  # Add a subtle text shadow for contrast
            }
        ),
        html.P(
            'Uncover the Hidden Layers of Earth with Intuitive Lithofacies Detection and Visualization',  # Add a description for the app
            style={
                'font-size': '1.5em',  # Set the font size for the description
                'color': 'white',  # Set the text color to white
                'text-align': 'center',  # Center the text horizontally
                'max-width': '800px',  # Set a maximum width for the text container
                'margin-top': '30px',  # Add some space above the text
                'text-shadow': '1px 1px 2px #000000'  # Add a subtle text shadow for contrast
            }
        )
    ]
)

