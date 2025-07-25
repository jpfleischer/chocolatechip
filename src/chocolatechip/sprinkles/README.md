This is a gui that puts a highlight on top of conflict videos to try and find
where to look where the conflict is happening.

To run it, you have to have chocolatechip installed. In order to do this,
you can cd to the top dir of chocolatechip (where the Makefile is) and then
use `make pip`

Once the library is installed, commands are available to you, such as
`chip sprinkles`. That command will look for MP4s in the directory where you're
standing and then reencode those clips to have helpful visual information.
(They will be saved in ~/sprinkle_output)

# Before
<img width="1041" height="731" alt="image" src="https://github.com/user-attachments/assets/84db0b7c-a984-48d6-9b37-3d274b795221" />

# After
<img width="1237" height="916" alt="image" src="https://github.com/user-attachments/assets/5659abbc-475e-43b4-9da8-19a8ab12656b" />


After generating sprinkles clips, `cd` to those clips (either by going
straight to ~/sprinkle_output or move those MP4s to another folder,
doesn't matter as long as the MP4s retain their filename structure)
then, as long as you are standing in the same dir as the clips,
use `chip sprinklesgui` command

<img width="800" height="765" alt="image" src="https://github.com/user-attachments/assets/27013430-9487-47f4-a8bc-69c86661a96c" />


Now you can mark videos as dangerous or harmless. Next, go to the `unflag` folder
and use that command (read the README there)
