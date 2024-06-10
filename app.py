import os
from flask import Flask, render_template, request
import penguin_model
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def basic():
    summary = None
    scatter_plot_url = None
    if request.method == 'POST':
        culmen_length = float(request.form['culmenlength'])
        culmen_depth = float(request.form['culmendepth'])
        flipper_length = float(request.form['flipperlength'])
        body_mass = float(request.form['bodymass'])
        y_pred = [[culmen_length, culmen_depth, flipper_length, body_mass]]
        trained_model = penguin_model.training_model()
        prediction_value = trained_model.predict(y_pred)[0]

        summary = {
            'Culmen Length': culmen_length,
            'Culmen Depth': culmen_depth,
            'Flipper Length': flipper_length,
            'Body Mass': body_mass
        }

        scatter_plot_url = create_scatter_plot(culmen_length, culmen_depth)

        result = determine_species(prediction_value)
        return render_template('index.html', result=result, summary=summary, scatter_plot_url=scatter_plot_url, ranges=get_species_ranges())
    
    return render_template('index.html', ranges=get_species_ranges())

def determine_species(prediction_value):
    chinstrap = 'This penguin is classified as Chinstrap.'
    adelie = 'This penguin is classified as Adélie.'
    gentoo = 'This penguin is classified as Gentoo.'

    if prediction_value == 'Chinstrap':
        return chinstrap
    elif prediction_value == 'Adelie':
        return adelie
    else:
        return gentoo

def create_scatter_plot(culmen_length, culmen_depth):
    df = pd.read_csv('penguins.csv')
    df.drop(columns=['island', 'sex'], inplace=True)
    df.rename(columns={'culmen_length_mm': 'culmen_length', 'culmen_depth_mm': 'culmen_depth'}, inplace=True)
    df = df.dropna()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='culmen_length', y='culmen_depth', hue='species', style='species')
    plt.scatter([culmen_length], [culmen_depth], color='red', marker='X', s=80, label='New Instance')
    plt.legend()
    plt.xlabel('Culmen Length (mm)')
    plt.ylabel('Culmen Depth (mm)')

    plot_path = os.path.join('static', 'scatter_plot.png')
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def get_species_ranges():
    df = pd.read_csv('penguins.csv')
    species_ranges = df.groupby('species').agg({
        'culmen_length_mm': ['min', 'max'],
        'culmen_depth_mm': ['min', 'max'],
        'flipper_length_mm': ['min', 'max'],
        'body_mass_g': ['min', 'max']
    })
    species_ranges.columns = ['_'.join(col).strip() for col in species_ranges.columns.values]
    return species_ranges.to_dict()

if __name__ == '__main__':
    app.run(debug=True)