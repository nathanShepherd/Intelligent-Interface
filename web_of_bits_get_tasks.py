orig = '''
bisect-angle: Find the line that bisects an angle evenly in two.
chase-circle: Keep your mouse inside a moving circle.
choose-date: Learn to operate a date picker tool.
choose-list: Choose an item from a drop down list.
circle-center: Find the center of a circle.
click-button: Click on a specific button in a generated form.
click-button-sequence: Click on buttons in a certain order.
click-checkboxes: Click desired checkboxes.
click-collapsible: Click a collapsible element to expand it.
click-collapsible-2: Find and click on a specified link, from collapsible elements.
click-color: Click the specified color.
click-dialog: Click the button to close the dialog box.
click-dialog-2: Click a specific button in a dialog box.
click-link: Click on a specified link in text.
click-menu-2: Find a specific item from a menu.
click-option: Click option boxes.
click-pie: Click items on a pie menu.
click-shades: Click the shades that match a specified color.
click-shape: Click on a specific shape.
click-tab: Click on a tab element.
click-tab-2: Click a link inside a specific tab element.
click-test: Click on a single button.
click-test-2: Click on one of two buttons.
click-widget: Click on a specific widget in a generated form.
count-shape: Count number of shapes.
count-sides: Count the number of sides on a shape.
find-midpoint: Find the shortest mid-point of two points.
focus-text: Focus into a text input.
focus-text-2: Focus on a specific text input.
grid-coordinate: Find the Cartesian coordinates on a grid.
guess-number: Guess the number.
identify-shape: Identify a randomly generated shape.
navigate-tree: Navigate a file tree to find a specified file or folder.
number-checkboxes: Draw a given number using checkboxes.
right-angle: Given two points, add a third point to create a right angle.
simon-says: Push the buttons in the order shown.
tic-tac-toe: Win a game of tic-tac-toe.
use-colorwheel: Use a color wheel.
use-colorwheel-2: Use a color wheel given specific random color.
use-slider: Use a slider to select a particular value.
use-slider-2: Use sliders to create a given combination.
use-spinner: Use a spinner to select given number.
'''

orig = orig.split('\n')
cleaned = []
for t in orig:
    out = t.split(':')
    cleaned.append(out[0])
    
for t in cleaned:
    print(t, end='\n')
