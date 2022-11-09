package com.deep.framework.ast.declaration;


import com.deep.framework.ast.Node;
import com.deep.framework.ast.expression.Name;

public class PackageDeclaration extends Declaration {
    private Name name;

    public PackageDeclaration(Node prarent) {
        super(prarent);
    }
}
