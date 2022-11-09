package com.deep.framework.ast.declaration;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.expression.Name;

public class FieldDeclaration extends Declaration {
    private String modifier;
    public Name name;

    public FieldDeclaration(Node prarent) {
        super(prarent);
    }
}
