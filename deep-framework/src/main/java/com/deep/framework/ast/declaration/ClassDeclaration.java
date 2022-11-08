package com.deep.framework.ast.declaration;

import java.util.List;

public class ClassDeclaration extends Declaration {

    private List<ClassDeclaration> extendedTypes;

    private List<InterfaceDeclaration> implementedTypes;
}
