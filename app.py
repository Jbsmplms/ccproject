from flask import Flask, render_template, request
import penguin_model
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])

def basic():
    if request.method == 'POST':
        culmen_length = request.form['culmenlength']
        culmen_depth = request.form['culmendepth']
        flipper_length = request.form['flipperlength']
        body_mass = request.form['bodymass']
        y_pred = [[culmen_length, culmen_depth, flipper_length, body_mass]]
        trained_model = penguin_model.training_model()
        prediction_value = trained_model.predict(y_pred)
        chinstrap = 'This penguin is classified as Chinstrap.'
        adelie = 'This penguin is classified as Adélie.'
        gentoo = 'This penguin is classified as Gentoo.'
        if prediction_value == 'Chinstrap':
            return render_template('index.html', chinstrap=chinstrap)
        elif prediction_value == 'Adelie':
            return render_template('index.html', adelie=adelie)
        else:
            return render_template('index.html', gentoo=gentoo) 
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)