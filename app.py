import os
from flask import Flask, render_template, request
import penguin_model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def basic():
    summary = None
    scatter_plot_url = None
    if request.method == 'POST':
        culmen_length = float(request.form['culmenlength'])
        culmen_depth = float(request.form['culmendepth'])
        flipper_length = float(request.form['flipperlength'])
        y_pred = [[culmen_length, culmen_depth, flipper_length]]
        trained_model = penguin_model.training_model()
        prediction_value = trained_model.predict(y_pred)[0]

        summary = {
            'Culmen Length': culmen_length,
            'Culmen Depth': culmen_depth,
            'Flipper Length': flipper_length
        }

        scatter_plot_url = create_scatter_plot(culmen_depth, culmen_length, flipper_length, include_new_instance=True)
        result = determine_species(prediction_value)
        return render_template('index.html', result=result, summary=summary, scatter_plot_url=scatter_plot_url, ranges=get_species_ranges())
    
    else:
        scatter_plot_url = create_scatter_plot(culmen_depth=None, culmen_length=None, flipper_length=None)
        return render_template('index.html', ranges=get_species_ranges(), scatter_plot_url=scatter_plot_url)
    
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

def create_scatter_plot(culmen_depth, culmen_length, flipper_length, include_new_instance=False):
    df = pd.read_csv('penguins.csv')
    df.drop(columns=['island', 'sex'], inplace=True)
    df.rename(columns={'culmen_length_mm': 'culmen_length', 'culmen_depth_mm': 'culmen_depth', 'flipper_length_mm': 'flipper_length'}, inplace=True)
    df = df.dropna()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    species = df['species'].unique()
    colors = ['r', 'b', 'g']
    color_dict = dict(zip(species, colors))

    for species in species:
        subset = df[df['species'] == species]
        ax.scatter(subset['culmen_depth'], subset['culmen_length'], subset['flipper_length'], 
                   color=color_dict[species], label=species)

    if include_new_instance:
        ax.scatter([culmen_depth], [culmen_length], [flipper_length], color='black', marker='X', s=100, label='New Instance')

    min_culmen_depth, max_culmen_depth = df['culmen_depth'].min(), df['culmen_depth'].max()
    min_culmen_length, max_culmen_length = df['culmen_length'].min(), df['culmen_length'].max()
    min_flipper_length, max_flipper_length = df['flipper_length'].min(), df['flipper_length'].max()

    if include_new_instance:
        min_culmen_depth = min(min_culmen_depth, culmen_depth)
        max_culmen_depth = max(max_culmen_depth, culmen_depth)
        min_culmen_length = min(min_culmen_length, culmen_length)
        max_culmen_length = max(max_culmen_length, culmen_length)
        min_flipper_length = min(min_flipper_length, flipper_length)
        max_flipper_length = max(max_flipper_length, flipper_length)

    ax.set_xlim([min_culmen_depth - 1, max_culmen_depth + 1])
    ax.set_ylim([min_culmen_length - 1, max_culmen_length + 1])
    ax.set_zlim([min_flipper_length - 1, max_flipper_length + 1])

    ax.set_xlabel('Culmen Depth (mm)')
    ax.set_ylabel('Culmen Length (mm)')
    ax.set_zlabel('Flipper Length (mm)')
    ax.legend()

    plot_path = os.path.join('static', 'scatter_plot.png')
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def get_species_ranges():
    df = pd.read_csv('penguins.csv')
    species_ranges = df.groupby('species').agg({
        'culmen_length_mm': ['min', 'max'],
        'culmen_depth_mm': ['min', 'max'],
        'flipper_length_mm': ['min', 'max']
    })
    species_ranges.columns = ['_'.join(col).strip() for col in species_ranges.columns.values]
    return species_ranges.to_dict()

if __name__ == '__main__':
    app.run(debug=True)