package com.deep.framework.ast.declaration;


import com.deep.framework.ast.Node;
import com.deep.framework.ast.expression.Name;
import com.deep.framework.ast.statement.Statement;

import static com.deep.framework.ast.lexer.TokenType.PACKAGE;

public class PackageDeclaration extends Declaration {
    private Name name;

    public PackageDeclaration(Node prarent) {
        super(prarent);
    }

    public static void parser(Node node) {
        if (node instanceof Statement && node.getChildrens().contains(PACKAGE)) {
            PackageDeclaration packageDeclare = new PackageDeclaration(node);
            packageDeclare.setChildrens(node.getChildrens());
            packageDeclare.getChildrens().remove(PACKAGE);
            node.getPrarent().replace(node, packageDeclare);
        }
    }
}
