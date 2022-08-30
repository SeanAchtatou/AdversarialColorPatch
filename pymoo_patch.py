
import numpy as np
import tensorflow as tf
import cv2
import csv
import math
import os
import imutils
import random
import time

from ast import literal_eval
from tensorflow import keras
from pymoo.problems.multi import Problem,ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_selection, get_termination
from pymoo.optimize import minimize

from model_Road import *

random.seed(1756)


p_form = ["p_circle","p_square","p_triangle","p_image"]
images_dir = "images_"
models_dir = "Models"
patch_dir = "patches"
result_dir = "Results"
color_main,color_mean_r,color_mean_g,color_mean_b = None,0,0,0
model_input_shape = 0
t_class_number = 0
t_class_U = None
no_target = None
numb_patch = None
form_ = None
image_small = None
model_Keras = True
original_prob = None
values = None

def folder_creation():
    path = os.listdir()
    try:
        if patch_dir not in path:
            print("\033[93m     Folder for the \033[4mpatches\033[0m \033[93m have been created.\033[0m")
            os.mkdir(patch_dir)
            for i in p_form:
                os.mkdir(f"{patch_dir}/{i}")
        else:
            print("\033[93m     Folder for \033[4mpatches\033[0m \033[93m already exists.  \033[0m")

    except:
        print("\033[91m An error occured. \033[0m")

def mean_patch(image):
    b,g,r = cv2.split(image)
    mean_r = int(np.mean(r))
    mean_g = int(np.mean(g))
    mean_b = int(np.mean(b))

    return mean_b,mean_g,mean_r

def mean(image):
    b,g,r = cv2.split(image)
    mean_r = int(np.mean(r))
    mean_g = int(np.mean(g))
    mean_b = int(np.mean(b))

    if form_ == "circle":
        temp_im = image.copy()
        radiusY = (temp_im.shape[1]/2)
        radiusX = (temp_im.shape[0]/2)
        y, x = np.ogrid[-radiusX: radiusX, -radiusY: radiusY]
        index = x**2 + y**2 > (radiusX-(temp_im.shape[0]/6))**2
        temp_im[:,:,:][index] = 0
        #cv2.imshow("P",temp_im)
        #cv2.waitKey(0)

        v = [np.unique(list(filter(([0,0,0]).__ne__,i.tolist())),return_counts=True,axis=0) for i in temp_im]

    elif form_ == "triangle":
        temp_im = image.copy()
        x,y,z = temp_im.shape
        grid = np.full((x,y),True)
        track = math.floor(y/2)
        track_ = math.floor(y/2)

        if x < y:
            jump = math.floor(x/(x-track))
        else:
            jump = math.ceil(x/(x-track))

        keep = jump
        tt = [track]
        for k in range(len(grid)):
            try:
                for p in tt:
                    grid[k][p] = False
            except:
                break
            jump -= 1
            if jump == 0:
                tt = [o for o in range(track-1,track_+2)]
                track = tt[0]
                track_ = tt[-1]
                jump = keep

        temp_im[:,:,:][grid] = 0
        v = [np.unique(list(filter(([0,0,0]).__ne__,i.tolist())),return_counts=True,axis=0) for i in temp_im]

    else:
        temp_im = image.copy()
        distX = math.ceil(image.shape[0]/12)
        distY = math.ceil(image.shape[1]/12)
        temp_im[0:distX] = 0
        temp_im[:,0:distY] = 0
        temp_im[image.shape[0]-distX:] = 0
        temp_im[:,image.shape[1]-distY:] = 0
        v = [np.unique(list(filter(([0,0,0]).__ne__,i.tolist())),return_counts=True,axis=0) for i in temp_im]



    dict = {}
    for i in range(len(v)):
        cc = len(v[i][0])
        for j in range(cc):
            d = str(v[i][0][j].tolist())
            m = v[i][1][j]
            if d in dict:
                dict[d] += m
            else:
                dict[d] = m

    best = literal_eval(max(dict,key=dict.get))
    mean_b, mean_g, mean_r = best

    if (mean_b > 150) and (mean_g > 150) and (mean_r > 150):
        return "W",mean_r,mean_g,mean_b
    if (mean_b < 50) and (mean_g < 50) and (mean_r < 50):
        return "E",mean_r,mean_g,mean_b
    if (mean_r >= mean_g) and (mean_r >= mean_b):
        return "R",mean_r,mean_g,mean_b
    if (mean_g >= mean_b) and (mean_g >= mean_r):
        return "G",mean_r,mean_g,mean_b
    if (mean_b >= mean_g) and (mean_b >= mean_r):
        return "B",mean_r,mean_g,mean_b


def calculate_diff(a,b):
    if no_target == "u":
        x = np.subtract(a,0)
        return x
    else:
        x = []
        for i in a:
            x.append(np.sum((i-b)**2))
        return x

    #cc = tf.keras.metrics.CategoricalCrossentropy()
    #c = [cc(i,b).numpy() for i in a]


def Algorithm(size_patch,t_class,model,image,form):
    global original_prob
    imageB = None
    if model_Keras:
        original_prob = np.argmax(model.predict(np.expand_dims(cv2.resize(image,model_input_shape),0)))
        imageB = image.copy()
    else:
        imageB = image_small
        cv2.imwrite('images_resized/resized_Image.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        test_ds = RoadDataset(pd.DataFrame([{'path':'images_resized/resized_Image.jpg'}])['path'],pd.DataFrame([{'bb':np.array([0,0,0,0])}])['bb'],pd.DataFrame([{'y':[0]}])['y'])
        x, y_class, y_bb = test_ds[0]
        xx = torch.FloatTensor(x[None,])
        out_class, out_bb = model(xx)
        pred = torch.max(out_class, 1)
        original_prob = np.array(pred[1])[0]

    x,y,z = imageB.shape
    n_var = ((size_patch**2)*3)+3
    last_ = (size_patch**2)*3

    patch_temp = np.reshape([0 for _ in range(last_)],(size_patch,size_patch,3)).astype(np.uint8)
    x_max,y_max,_ = imutils.rotate_bound(patch_temp,45).shape

    add_main = None
    add_main_ = None

    if color_main == "W":
        channel_c = 3
        #add_main = min(color_mean_r,max(color_mean_b,color_mean_g)+ 50)
        add_main = max(0,color_mean_r-50)
        add_main_ = min(255,color_mean_r+40)
    if color_main == "E":
        channel_c = 3
        #add_main = min(color_mean_r,max(color_mean_b,color_mean_g)+ 50)
        add_main = max(0,color_mean_r-50)
        add_main_ = min(255,color_mean_r+40)
    if color_main == "R":
        channel_c = 3
        #add_main = min(color_mean_r,max(color_mean_b,color_mean_g)+ 50)
        add_main = max(0,color_mean_r-50)
        add_main_ = min(255,color_mean_r+100)
    if color_main == "G":
        channel_c = 2
        #add_main = min(color_mean_g,max(color_mean_b,color_mean_r)+ 50)
        add_main = max(0,color_mean_g-50)
        add_main_ = min(255,color_mean_g+100)
    if color_main == "B":
        channel_c = 1
        #add_main = min(color_mean_b,max(color_mean_r,color_mean_g)+ 50)
        add_main = max(0,color_mean_b-50)
        add_main_ = min(255,color_mean_b+100)


    #x_l = [add_main if (((i+1)%3)-channel_c)%3==0 else 0 for i in range(n_var)]
    x_l = []
    for i in range(n_var):
        if (((i+1)%3)-channel_c)%3==0:
            x_l.append(add_main)
        else:
            if (((i+1)%3)-1)%3==0:
                x_l.append(max(0,color_mean_b-30))
            if (((i+1)%3)-2)%3==0:
                x_l.append(max(0,color_mean_g-30))
            if (((i+1)%3)-3)%3==0:
                x_l.append(max(0,color_mean_r-30))



    x_l[-1] = math.ceil(image.shape[0]/12)
    x_l[-2] = math.ceil(image.shape[1]/12)
    x_l[-3] = 0
    x_u = []
    for i in range(n_var):
        if i == (last_):
            x_u.append(x-x_max-math.ceil(image.shape[0]/12))
        if i == (last_+1):
            x_u.append(y-y_max-math.ceil(image.shape[1]/12))
        if i == (last_+2):
            x_u.append(360)
        if (i < last_):
            if (((i+1)%3)-channel_c)%3==0:
                x_u.append(add_main_)
            else:
                if (((i+1)%3)-1)%3==0:
                    x_u.append(min(255,color_mean_b+30))
                if (((i+1)%3)-2)%3==0:
                    x_u.append(min(255,color_mean_g+30))
                if (((i+1)%3)-3)%3==0:
                    x_u.append(min(255,color_mean_r+30))




    class MyProblem(Problem):
        def __init__(self):
            super().__init__(n_var=n_var,
                             n_obj=1,
                             n_constr=0,
                             xl=x_l,
                             xu=x_u)

        def _evaluate(self, X, out, *args, **kwargs):
            global t_class_U, original_prob

            patches_ = X[:,:-3]
            heights_ = X[:,-3]
            widths_ = X[:,-2]
            angles_ = X[:,-1]
            patches = np.reshape(patches_,(len(X),size_patch,size_patch,3)).astype(np.uint8)

            if form == "circle":
                radius = patches.shape[1]/2
                y, x = np.ogrid[-radius: radius, -radius: radius]
                index = x**2 + y**2 > radius**2
                for i in range(len(patches)):
                    patches[i][:,:,:][index] = 0

            if form == "triangle":
                x = patches.shape[1]
                grid = np.full((x,x),True)
                track = math.floor(x/2)
                track_ = math.floor(x/2)
                jump = 2

                keep = jump
                tt = [track]
                for k in range(len(grid)):
                    try:
                        for p in tt:
                            grid[k][p] = False
                    except:
                        break
                    jump -= 1
                    if jump == 0:
                        tt = [o for o in range(track-1,track_+2)]
                        track = tt[0]
                        track_ = tt[-1]
                        jump = keep
                for i in range(len(patches)):
                    patches[i][:,:,:][grid] = 0


            final_image = []
            for i in range(len(patches)):
                OTarget = imageB.copy()
                im = imutils.rotate_bound(patches[i],angles_[i])
                x,y,_ = im.shape
                part = OTarget[heights_[i]:heights_[i]+x,widths_[i]:widths_[i]+y]
                w = im == (0,0,0)
                im[w] = part[w]
                OTarget[heights_[i]:heights_[i]+x,widths_[i]:widths_[i]+y] = im
                cv2.imshow("W",OTarget)
                cv2.waitKey(1)
                if model_Keras:
                    OTarget = cv2.resize(OTarget,model_input_shape)
                else:
                    fimage = image.copy()
                    fimage[values[0]:values[2],values[1]:values[3]] = OTarget.copy()
                    #cv2.imshow("W",fimage)
                    #cv2.waitKey(1)
                    OTarget = fimage.copy()

                final_image.append(OTarget)

            if model_Keras:
                pred = model.predict(np.array(final_image))
                print(f"\nClass predicted: \n {pred.argmax(1)[:5]}")
                print(f"Probability of highest class: \n {pred[np.array([i for i in range(len(pred))][:5]),pred.argmax(1)[:5]]}")
                print(f"Probability of the class of the original image: \n {pred[:,original_prob][:5]}")
            else:
                ff = []
                for p in final_image:
                    #cv2.imwrite('images_resized/resized_Image.jpg', cv2.cvtColor(p, cv2.COLOR_RGB2BGR))
                    #test_ds = RoadDataset(pd.DataFrame([{'path':'images_resized/resized_Image.jpg'}])['path'],pd.DataFrame([{'bb':np.array([0,0,0,0])}])['bb'],pd.DataFrame([{'y':[0]}])['y'])
                    #o, _, _ = test_ds[0]

                    p = p/255
                    ffI = create_mask(np.array([0,0,0,0]), p)
                    ffI , _ = center_crop(p), center_crop(ffI)
                    ffI  = normalize(ffI)
                    ffI  = np.rollaxis(ffI, 2)
                    ff.append(ffI)

                xx = torch.FloatTensor(np.array(ff))

                out_class = torch.FloatTensor([])
                for pp in range(0,pop_size,64):
                    ppp = xx[pp:pp+64, :, :]
                    out_class = torch.cat((out_class,model(ppp)[0]),0)

                pred = torch.max(out_class, 1)[1]
                prob = F.softmax(out_class,dim=1)
                print(f"\nClass predicted: \n {pred[:5]}")
                print(f"Probability of highest class: \n {[float(max(prob[i])) for i in range(len(prob))[:5]]}")
                print(f"Probability of the class of the original image: \n {[float(prob[i][original_prob]) for i in range(len(prob))[:5]]}")


            if no_target == "nc":
                best_prob = 0
                best_class = 0
                if model_Keras:
                    indexes = pred.argmax(1)
                    for o in range(len(pred)):
                        if (indexes[o] != original_prob) and (pred[o][indexes[o]] > best_prob):
                            best_prob = pred[o][indexes[o]]
                            best_class = indexes[o]

                    t_class_U = np.zeros(len(pred[1]))
                    t_class_U[best_class] = 1
                else:
                    indexes = pred
                    pred = [float(max(prob[i])) for i in range(len(prob))]
                    for o in range(len(pred)):
                        if (indexes[o] != original_prob) and (pred[o] > best_prob):
                            best_prob = pred[o]
                            best_class = indexes[o]

                    t_class_U = np.zeros(len(pred))
                    t_class_U[best_class] = 1

                f1 = calculate_diff(pred,t_class_U)
                out["F"] = np.column_stack([f1])

            if no_target == "u":
                pred = [o[original_prob] for o in pred]
                f1 = calculate_diff(pred,t_class)
                out["F"] = np.column_stack([f1])

            if no_target == "":
                f1 = calculate_diff(pred,t_class)
                out["F"] = np.column_stack([f1])



    vectorized_problem = MyProblem()
    pop_size = 128
    algorithm = GA(
        pop_size=pop_size,
        n_offsprings=128,
        selection=get_selection("random"),
        sampling=get_sampling("int_random"),
        crossover=get_crossover("int_sbx", prob=0.9, eta=15),
        mutation=get_mutation("int_pm", eta=20),
        eliminate_duplicates=True
    )

    termination = get_termination("n_gen", 200)

    res = minimize(vectorized_problem,
                   algorithm,
                   termination,
                   seed=3746,
                   save_history=True,
                   verbose=True,
                   return_least_infeasible=True)


    pop = res.pop
    print("Best solutions found: \nX = %s\nF = %s" % (pop.get("X")[:numb_patch], pop.get("F")[:numb_patch]))

    try:
        for i in range(numb_patch):
            image_flat = pop.get("X")[i]
            patch_ = image_flat[:-3]
            height_ = image_flat[-3]
            width_ = image_flat[-2]
            angle_ = image_flat[-1]
            patch = np.reshape(patch_,(size_patch,size_patch,3)).astype(np.uint8)

            if form == "circle":
                radius = patch.shape[0]/2
                y, x = np.ogrid[-radius: radius, -radius: radius]
                index = x**2 + y**2 > radius**2
                patch[:,:,:][index] = 0

            if form == "triangle":
                x = patch.shape[1]
                grid = np.full((x,x),True)
                track = math.floor(x/2)
                track_ = math.floor(x/2)
                jump = 2

                keep = jump
                tt = [track]
                for k in range(len(grid)):
                    try:
                        for p in tt:
                            grid[k][p] = False
                    except:
                        break
                    jump -= 1
                    if jump == 0:
                        tt = [o for o in range(track-1,track_+2)]
                        track = tt[0]
                        track_ = tt[-1]
                        jump = keep

                patch[:,:,:][grid] = 0

            OTarget = image.copy()
            im = imutils.rotate_bound(patch,angle_)
            x,y,_ = im.shape
            part = OTarget[height_:height_+x,width_:width_+y]
            w = im == (0,0,0)
            im[w] = part[w]
            OTarget[height_:height_+x,width_:width_+y] = im
            OTarget_M = cv2.resize(OTarget,model_input_shape)
            prob_final = model.predict(np.expand_dims(OTarget_M,0))
            class_prob = prob_final[0].argmax(0)
            prob = prob_final[0][class_prob]
            cv2.imwrite(f"{patch_dir}/p_{form}/final_patch_{size_patch}S_{height_}X{width_}Y_{angle_}A_{int(time.time())}T.png",patch)
            cv2.imwrite(f"{patch_dir}/{p_form[-1]}/patch_image_{size_patch}S_{height_}X{width_}Y_{angle_}A_{int(time.time())}T.png",OTarget)
            cv2.imshow(f"Best patch for size-{size_patch}",patch)
            cv2.imshow(f"Position of best patch in Image",OTarget)
            print(f"Classes/Prob predicted by the model with the best patch: {class_prob},{prob}")

            stats = [size_patch,height_,width_,angle_,class_prob,prob,original_prob,prob_final[0][original_prob]]
            with open(f"{result_dir}/Results_{form}_{no_target}.csv","a",newline="") as f:
                write = csv.writer(f)
                write.writerow(stats)
                f.close()

            cv2.waitKey(1)
    except:
        print("No solution found here.")


def main(image,model,t_class,form):
    global color_main, t_class_U, color_mean_r, color_mean_g, color_mean_b, model_input_shape, image_small, values
    x,y,z = image.shape

    if model_Keras:
        color, mean_r, mean_g, mean_b = mean(image)
    else:
        df_train = generate_train_df(annotations_path)
        df_train.head()
        class_dict = {'speedlimit': 0, 'stop': 1, 'crosswalk': 2, 'trafficlight': 3}
        df_train['class'] = df_train['class'].apply(lambda x:  class_dict[x])
        new_paths = []
        new_bbs = []
        train_path_resized = Path('./images')
        for index, row in df_train.iterrows():
            new_path,new_bb = resize_image_bb(row['filename'], train_path_resized, create_bb_array(row.values),300)
            new_paths.append(new_path)
            new_bbs.append(new_bb)
        df_train['new_path'] = new_paths
        df_train['new_bb'] = new_bbs
        image_index = [i for i,j in enumerate(df_train['filename']) if images[image_selected] in str(j)]
        values = df_train.values[image_index][0][9].astype(int)
        image_small = image[values[0]:values[2],values[1]:values[3]]
        x,y,z = image_small.shape
        color, mean_r, mean_g, mean_b = mean(image_small)
    color_main,color_mean_r,color_mean_g,color_mean_b = color,mean_r,mean_g,mean_b
    sizes_patches = []
    for i in range(3,10,1):
        b = int(x/i)
        if b%2 == 1:
            b += 1
        sizes_patches.append(b)

    header = ["Patch_Size","PositionX","PositionY","Rotation","Class_Attack","Probability","Class_Original","Final_Probability"]
    with open(f"{result_dir}/Results_{form}_{no_target}.csv","w",newline="") as f:
        write = csv.DictWriter(f,fieldnames=header)
        write.writeheader()
        f.close()

    for i in sizes_patches:
        Algorithm(i,t_class,model,image,form)


if "__main__" == __name__:
    print("\033[1;33;40m ################################# ADVERSARIAL PATCH WITH COLOR RESTRICTION #################################### \033[0m \n")

    while True:
        try:
            model_selected = input("\nType of Model (Keras,PyTorch) >")
            if (model_selected == "Keras" or model_selected == "PyTorch"):
                print(f"\033[1;32;40m{model_selected}\033[0m has been selected.")
                if model_selected != "Keras":
                    model_Keras = False
                break
            else:
                print("\033[91m     Please, enter a correct value. \033[0m")
        except:
            print("\033[91m     Please, enter a correct value. \033[0m")

    images = [i for i in os.listdir(images_dir)]
    if len(images) < 1:
        print(f"\033[91m     None image have been found. Please insert an image in the corresponding folder {images_dir}. \033[0m")
        exit()
    else:
        print("\033[4m\033[2;37;35mImages found:\033[0m")
        count = 0
        for i in images:
            print(f"    [\033[32m{count}\033[0m] {i}")
            count += 1

    while True:
        try:
            image_selected = int(input("\nEnter the number of the Image to attack (should be a number) >"))
            image = cv2.imread(os.path.join(images_dir,images[image_selected]))
            print(f"\033[1;32;40m{images[image_selected]}\033[0m has been selected.")
            break
        except:
            print("\033[91m     Please, enter a correct number. \033[0m")

    models = [i for i in os.listdir(models_dir)]
    if len(models) < 1:
        print(f"\033[91m     None model have been found. Please insert a model in the corresponding folder {models_dir}. \033[0m")
        exit()
    else:
        print("\n\033[4m\033[2;37;35mModels found:\033[0m")
        count = 0
        for i in models:
            print(f"    [\033[32m{count}\033[0m] {i}")
            count += 1

    while True:
        try:
            model_selected = int(input("\nEnter the number of the Model to use (should be a number) >"))

            if model_Keras:
                model = keras.models.load_model(f"{models_dir}/{models[model_selected]}")
                config = model.get_config()
                model_input_shape = config["layers"][0]["config"]["batch_input_shape"][1:3]
            else:
                model = torch.load(f"{models_dir}/{models[model_selected]}")
                model_input_shape = (3,1.49*300,300) #CHECK TO GET INPUT FROM MODEL

            print(f"\033[1;32;40m{models[model_selected]}\033[0m has been selected.")

            break
        except:
            print("\033[91m     Please, enter a correct number. \033[0m")

    while True:
        try:
            if model_Keras:
                numb = model.output_shape[-1]
            else:
                numb = 4

            print(f"\n\033[4m\033[2;37;35mNumber of labels of the model:\033[0m \n {numb}")
            t_class = input("\nClass to Attack (target attack) [NUMBER]/ No Class (best class to attack) [NC] / Untarget Attack [U] >").lower()
            if t_class == "nc":
                print(f"\033[1;32;40mBest Target\033[0m has been selected.")
                t_class_arr = t_class
                no_target = t_class
                break

            if t_class == "u":
                print(f"\033[1;32;40mNo Target\033[0m has been selected.")
                t_class_arr = t_class
                no_target = t_class
                break

            if t_class == "":
                print("\033[91m     Please, enter a correct option. \033[0m")

            else:
                print(f"\033[1;32;40m{t_class}\033[0m has been selected.")
                t_class_arr = np.zeros(numb)
                t_class_arr[int(t_class)] = 1
                t_class_number = int(t_class)
                no_target = ""
                break
        except:
            print("\033[91m     Please, enter a correct number. \033[0m")

    while True:
        try:
            form = input("\nForm of the adversarial patches (circle,square,triangle) >")
            if (form == "circle") or (form == "square") or (form == "triangle"):
                break
        except:
            print("\033[91m     Please, enter a correct form. \033[0m")

    print(f"\033[1;32;40m{form}\033[0m has been selected.")

    while True:
        try:
            form_ = input("\nShape of the traffic sign to attack (circle,triangle,square) >")
            if (form_ == "circle") or (form_ == "triangle") or (form_ == "square"):
                break
        except:
            print("\033[91m     Please, enter a correct form. \033[0m")

    print(f"\033[1;32;40m{form_}\033[0m has been selected.")

    while True:
        try:
            nb_patch = int(input("\nAmount of adversarial patches generated (<= 10) >"))
            if (1 <= nb_patch <= 10):
                numb_patch = nb_patch
                break
            else:
                print("\033[91m     Please, enter a correct number. \033[0m")

        except:
            print("\033[91m     Please, enter a correct value. \033[0m")

    print(f"\033[1;32;40m{nb_patch}\033[0m adversarial patches has been selected.")

    folder_creation()

    forms = ["circle","square","triangle"]

    for form in forms:
        main(image,model,t_class_arr,form)
