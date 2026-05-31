#!/usr/bin/env bash

# shellcheck disable=SC2034

# Nominal minimum heap size (MB) for default size configuration (with compressed pointers)
declare -rgA MIN_HEAP=(
    [batik]=185
    [biojava]=97
    [cassandra]=104
    [fop]=17
    [h2]=707
    [h2o]=58
    [jme]=29
    [jython]=27
    [kafka]=209
    [luindex]=38
    [lusearch]=21
    [pmd]=99
    [spring]=52
    [tomcat]=17
    [zxing]=75
)
