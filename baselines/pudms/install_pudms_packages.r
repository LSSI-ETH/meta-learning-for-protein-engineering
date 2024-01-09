url <- "https://cran.r-project.org/src/contrib/Archive/PUlasso/PUlasso_3.2.4.tar.gz"
pkgFile <- "PUlasso_3.2.4.tar.gz"
download.file(url = url, destfile = pkgFile)
# Expand the zip file using whatever system functions are preferred
untar("PUlasso_3.2.4.tar.gz")
# Install package
install.packages(pkgs=pkgFile, type="source", repos=NULL)
# Delete package tarball
unlink(pkgFile)

if(!require(remotes)) install.packages("remotes")
remotes::install_github("RomeroLab/pudms")
quit(save="no")

#if the installation of pudms fails, it may be related to https://github.com/r-lib/remotes/issues/210
#workaround is to use devtools. uncomment the following line
#devtools::install_github('RomeroLab/pudms')

install.packages("stringr")

install.packages("dplyr")
