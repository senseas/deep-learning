package com.deep.framework.ast.statement;

import lombok.Data;

import java.util.ArrayList;
import java.util.List;

@Data
public class FunctionProduction {

    List<String> functionDecl = new ArrayList() {{
        add("'public'|'private'|'protected'|'static'|'abstract'|'native'|'strictfp'|'synchronized' r(0,1)");
        add("->'final' r(0,1)");
        add("->'void'|TypeDecl");
        add("->identifier");
        add("->(typeDecl->arDeclId)");
        add("->block");
    }};

}
