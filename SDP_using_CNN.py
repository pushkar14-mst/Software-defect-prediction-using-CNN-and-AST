import tkinter
import customtkinter
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense

# CNN MODEL
data = pd.read_csv("F:/BE-Major-Project/Defect_dataset.csv").replace(np.nan, 0)
data = data.astype("float32")
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
    x, y, test_size=0.20, random_state=7
)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

model = Sequential()
model.add(Dense(4, input_shape=(19,), activation="relu"))
model.add(Dense(4, activation="relu"))
model.add(Dense(4, activation="tanh"))
model.add(Dense(1, activation="sigmoid"))

# FEATURE EXTRACTION USING AST
class WMCVisitor(ast.NodeVisitor):
    def __init__(self):
        self.wmc = 0

    def visit_ClassDef(self, node):
        wmc = 0
        for n in ast.walk(node):
            if isinstance(n, ast.FunctionDef):
                wmc += 1
        node.wmc = wmc
        self.wmc += wmc
        self.generic_visit(node)


def calculate_cbo(code):
    class CBOVisitor(ast.NodeVisitor):
        def __init__(self):
            self.cbo = 0
            self.current_class = None
            self.classes_referenced = set()

        def visit_ClassDef(self, node):
            self.current_class = node
            self.generic_visit(node)
            self.current_class = None

        def visit_Name(self, node):
            if self.current_class is not None and node.id in self.classes_referenced:
                self.cbo += 1

        def visit_Attribute(self, node):
            if self.current_class is not None and isinstance(node.value, ast.Name):
                self.classes_referenced.add(node.value.id)

    parsed_code = ast.parse(code)
    cbo_visitor = CBOVisitor()
    cbo_visitor.visit(parsed_code)
    cbo = cbo_visitor.cbo
    return cbo


def extract_loc(code):
    module = ast.parse(code)
    loc = sum(1 for node in ast.walk(module) if isinstance(node, ast.stmt))
    return loc


def extract_lcom(code):
    module = ast.parse(code)
    methods = [node for node in ast.walk(module) if isinstance(node, ast.FunctionDef)]
    lcom = sum(len(set(ast.walk(method))) > 1 for method in methods)
    return lcom


def extract_cbo(code):
    module = ast.parse(code)
    classes = {
        node.name: set() for node in ast.walk(module) if isinstance(node, ast.ClassDef)
    }
    for node in ast.walk(module):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
        ):
            caller = None
            for ancestor in ast.walk(node):
                if isinstance(ancestor, ast.ClassDef):
                    caller = ancestor.name
                    break
            callee = node.func.value.id
            if caller and callee in classes:
                classes[caller].add(callee)
    cbo = sum(len(callees) for callees in classes.values())
    return cbo


def extract_wmc(code):
    module = ast.parse(code)
    classes = {
        node.name: 0 for node in ast.walk(module) if isinstance(node, ast.ClassDef)
    }
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef):
            for ancestor in ast.walk(node):
                if isinstance(ancestor, ast.ClassDef):
                    classes[ancestor.name] += 1
    wmc = sum(classes.values())
    return wmc


def extract_dam(code):
    module = ast.parse(code)
    variable_refs = {node.id for node in ast.walk(module) if isinstance(node, ast.Name)}
    attribute_refs = {
        node.attr for node in ast.walk(module) if isinstance(node, ast.Attribute)
    }
    dam = len(attribute_refs) / (len(variable_refs) + len(attribute_refs))
    return dam


def extract_ce(code):
    module = ast.parse(code)
    imports = {
        node.names[0].name for node in ast.walk(module) if isinstance(node, ast.Import)
    }
    for node in ast.walk(module):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if node.module:
                    imports.add(f"{node.module}.{alias.name}")
                else:
                    imports.add(alias.name)

    ce = len(imports)
    return ce


def extract_moa(code):
    module = ast.parse(code)
    functions = [
        node
        for node in ast.walk(module)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    class_defs = {
        node.name for node in ast.walk(module) if isinstance(node, ast.ClassDef)
    }
    module_functions = [
        node.name
        for node in functions
        if not any(isinstance(parent, ast.ClassDef) for parent in ast.walk(node))
    ]
    moa = len(set(class_defs)) / (len(class_defs) + len(module_functions))
    return moa


def extract_npm(code):
    module = ast.parse(code)
    functions = [
        node
        for node in ast.walk(module)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    classes = [node for node in ast.walk(module) if isinstance(node, ast.ClassDef)]
    public_methods = [
        node for node in functions + classes if not node.name.startswith("_")
    ]
    npm = len(public_methods)

    return npm


def extract_max_cc(code):

    module = ast.parse(code)

    max_cc = 1

    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            cc = calc_cc(node)
            if cc > max_cc:
                max_cc = cc

    return max_cc


def calc_cc(node):
    cc = 1
    for subnode in ast.walk(node):
        if (
            isinstance(subnode, ast.If)
            or isinstance(subnode, ast.For)
            or isinstance(subnode, ast.While)
            or isinstance(subnode, ast.With)
            or isinstance(subnode, ast.AsyncWith)
        ):
            cc += 1
    return cc


def extract_cc(node):
    cc = 1
    for child in ast.iter_child_nodes(node):
        if isinstance(
            child,
            (
                ast.If,
                ast.For,
                ast.While,
                ast.Try,
                ast.With,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):
            cc += 1
    return cc


def extract_v(node):
    # Count the number of vertices in the control flow graph of the node
    v = 1
    for child in ast.iter_child_nodes(node):
        v += extract_v(child)
    return v


def extract_ev(node):
    # Count the number of edges in the control flow graph of the node
    ev = 0
    for child in ast.iter_child_nodes(node):
        ev += 1 + extract_ev(child)
    return ev


def extract_iv(node):
    # Count the number of independent paths in the control flow graph of the node
    iv = 1
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.If, ast.For, ast.While, ast.Try, ast.With)):
            iv += extract_iv(child)
    return iv


def extract_statements(node):
    # Count the number of statements in the node
    return sum(1 for n in ast.walk(node) if isinstance(n, ast.stmt))


def extract_variables(node):
    # Count the number of variables in the node
    return sum(1 for n in ast.walk(node) if isinstance(n, ast.Name))


def extract_decisions(node):
    # Count the number of decisions in the node
    return sum(1 for n in ast.walk(node) if isinstance(n, ast.If))


def extract_loops(code):
    count = 0
    for node in ast.walk(code):
        if isinstance(node, ast.For) or isinstance(node, ast.While):
            count += 1
    return count


def count_functions(node):
    if isinstance(node, ast.FunctionDef):
        return 1

    count = 0
    for child_node in ast.iter_child_nodes(node):
        count += count_functions(child_node)

    return count


def count_assignments(node):
    """Recursively count the number of assignments in the given AST."""
    if isinstance(node, ast.Assign):
        return 1
    count = 0
    for child_node in ast.iter_child_nodes(node):
        count += count_assignments(child_node)

    return count


def extract_dit(source_code):
    tree = ast.parse(source_code)
    class_defs = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    dit_dict = {class_def.name: 0 for class_def in class_defs}

    for class_def in class_defs:
        base_classes = [base for base in class_def.bases if isinstance(base, ast.Name)]
        base_names = [base.id for base in base_classes] if base_classes else []
        for base_name in base_names:
            dit_dict[base_name] = max(dit_dict[base_name], dit_dict[class_def.name] + 1)

    return max(dit_dict.values())


def extract_noc(source_code):
    tree = ast.parse(source_code)
    class_defs = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    noc = 0
    for class_def in class_defs:
        class_bases = [base for base in class_def.bases if isinstance(base, ast.Name)]
        if class_bases:
            noc += len(class_bases)
    return noc


customtkinter.set_appearance_mode("System")  # Modes: system (default), light, dark
customtkinter.set_default_color_theme(
    "blue"
)  # Themes: blue (default), dark-blue, green

app = customtkinter.CTk()
window = app.geometry("1080x720")

app.title("Software Defect Prediction Using CNN")
frame_1 = customtkinter.CTkFrame(master=app)
frame_1.pack(pady=10, padx=30, fill="both", expand=True)


def get_code():
    inp = textbox.get("1.0", "end-1c")
    code = f"""{inp}"""
    parsed_code = ast.parse(code)
    wmc_visitor = WMCVisitor()
    wmc_visitor.visit(parsed_code)
    wmc = wmc_visitor.wmc
    cbo = calculate_cbo(parsed_code)
    loc = extract_loc(parsed_code)
    lcom = extract_lcom(parsed_code)
    cbo = extract_cbo(parsed_code)
    wmc = extract_wmc(parsed_code)
    dam = extract_dam(parsed_code)
    ce = extract_ce(parsed_code)
    moa = extract_moa(parsed_code)
    npm = extract_npm(parsed_code)
    max_cc = extract_max_cc(parsed_code)
    cc = extract_cc(parsed_code)
    extracted_statements = extract_statements(parsed_code)
    extracted_variables = extract_variables(parsed_code)
    extracted_decisions = extract_decisions(parsed_code)
    ev = extract_ev(parsed_code)
    iv = extract_iv(parsed_code)
    extracted_loops = extract_loops(parsed_code)
    rfc = count_functions(parsed_code)
    ca = count_assignments(parsed_code)
    # dit=extract_dit(code)
    noc = extract_noc(code)

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(X_train, Y_train, epochs=100, verbose=0)

    input_data = sc.transform(
        np.array(
            [
                [
                    ev,
                    iv,
                    wmc,
                    cbo,
                    loc,
                    lcom,
                    dam,
                    ce,
                    moa,
                    npm,
                    max_cc,
                    cc,
                    extracted_statements,
                    extracted_decisions,
                    extracted_variables,
                    extracted_loops,
                    rfc,
                    ca,
                    noc,
                ]
            ]
        )
    )
    prediction = model.predict(input_data)
    print(prediction)
    if prediction >= 0.5:
        print("There is a bug possiblity, needed to be redesigned")
        label_3.configure(text="There is a bug possiblity, needed to be redesigned")
    else:
        print("No Bugs Predicted!!!")
        label_3.configure(text="No Bugs Predicted!!!")


label_1 = customtkinter.CTkLabel(
    master=frame_1,
    text="Predictor",
    font=customtkinter.CTkFont(size=20, weight="bold"),
    justify=customtkinter.CENTER,
)
label_1.pack(pady=10, padx=10)
button_1 = customtkinter.CTkButton(
    master=frame_1, text="Predict Bugs", command=get_code
)
button_1.pack(pady=20, padx=20)
textbox = customtkinter.CTkTextbox(master=frame_1, width=450, height=720)
textbox.pack(padx=20, pady=20, fill="both", expand=True)

# suggestions box
frame_2 = customtkinter.CTkFrame(master=app, width=450)
frame_2.pack_propagate(0)
frame_2.pack(side="bottom", padx=20, pady=20, anchor=tkinter.CENTER)
label_2 = customtkinter.CTkLabel(
    master=frame_2,
    text="Suggestions",
    font=customtkinter.CTkFont(size=20, weight="bold"),
    justify=customtkinter.LEFT,
)
label_2.pack(pady=10, padx=10)
label_3 = customtkinter.CTkLabel(
    master=frame_2,
    text="Please paste your code for suggestions!!",
    font=customtkinter.CTkFont(size=20),
    justify=customtkinter.LEFT,
)
label_3.pack(pady=10, padx=10)


app.mainloop()
