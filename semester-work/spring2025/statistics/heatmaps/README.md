i want you to clone

pip install git+<insert git repo link to chocolatechip> [FINISHED]

in a docker contianer.

make sure that it installs without any errors. 
if there are errors, we have to fix them
in the chocolatechip repository.    [FINISHED]

after you install it,
look into docker secrets. 

we want to learn how to include environment variables
so that we can include the MySQL credentials.

once you do that,
we should be able to automatically make heatmaps.
in that docker container.

# 2/4

once it compiles, and you are able to see the
environment variables with the credentials,
you have to change the sourcce code to not use
login.yaml.

docker secrets in the run command [FINISHED]

in order to make these source code changes to begin with,
you have to make a new branch on chocolatechip.

you could theoretically do a volume mount 
-v ../../src:src

DONT git clone anymore. just do the volume mount.   [FINISHED]


...

instead, cohcolatechip has to use os.env it has to use the environment
variables to login and make the connection with
the mysql database. you have to go to mysqlconnector and
adjust the logic to stop using YAML and start using env.

Pass in ENV variables directly into the docker container    [FINISHED]