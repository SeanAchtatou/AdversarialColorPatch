# Adversarial Patches with Color Restriction

<p align="center">
  <img src="https://github.com/SeanAchtatou/Adversarial_Patches_With_Color_Restriction/blob/main/Pictures/menu.PNG">
</p>

This Project have been developped in the University of Luxembourg Security and Trust (SnT). This attack allows to apply the basic adversarial attacks on "Road signs" with patches of different forms (square, circle) and sizes (33% up to 10% of the original image) with an additional restriction based on the color of the attacked road sign in order to minimize the detection of the patch by human.

## Description

This application outputs the most efficient patches in order to attacks road signs. You have the possibility between three options of attacks: <br />
<br />
-: Target Attack (give the corresponding road sign to attack and create the most efficient patch) <br />
-: Untarget Attack (create the most efficient patch allowing to produce the lowest probability of the original label) <br />
-: No Label (create the most efficient patch based on the best label probability) <br />


### Dependencies

* tensorflow
* pymoo
* numpy

### Executing program

-: At each launch, the application will fetch the available image to attack, and will ask to select one of them. <br />

<p align="center">
  <img  src="https://github.com/SeanAchtatou/Adversarial_Patches_With_Color_Restriction/blob/main/Pictures/images.PNG">
  <img src="https://github.com/SeanAchtatou/Adversarial_Patches_With_Color_Restriction/blob/main/Pictures/pedes.PNG">
</p>
<br />

Once selected, you have the possibility to choose between the three options of attacks.<br />
-: The target attack is based on the number of the corresponding class of the road signs (displayed on console), simply choose one of the number and it will continue. <br />
-: NC (no label)  <br />
-: U (untarget attack) <br />
<br />

<p align="center">
  <img src="https://github.com/SeanAchtatou/Adversarial_Patches_With_Color_Restriction/blob/main/Pictures/road.PNG">
</p>
<br />

Finally, you can select between the types of patch to apply such as square or circle.<br />

<p align="center">
  <img src="https://github.com/SeanAchtatou/Adversarial_Patches_With_Color_Restriction/blob/main/Pictures/crircle.PNG">
</p>


## Authors

ACHTATOU Sean <br />
sean.achtatou@hotmail.be
