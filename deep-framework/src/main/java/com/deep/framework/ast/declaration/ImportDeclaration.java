package com.deep.framework.ast.declaration;


import com.deep.framework.ast.Node;
import com.deep.framework.ast.expression.Name;

public class ImportDeclaration extends Declaration {
    private Name name;

    public ImportDeclaration(Node prarent) {
        super(prarent);
    }
}
