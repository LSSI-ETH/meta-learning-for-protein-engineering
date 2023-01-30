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

install.packages("stringr")

install.packages("dplyr")

