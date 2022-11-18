package com.deep.framework.ast.declaration;


import com.deep.framework.ast.Node;
import com.deep.framework.ast.expression.Name;
import com.deep.framework.ast.statement.Statement;

import static com.deep.framework.ast.lexer.TokenType.IMPORT;

public class ImportDeclaration extends Declaration {
    private Name name;

    public ImportDeclaration(Node prarent) {
        super(prarent);
    }

    public static void parser(Node node) {
        if (node instanceof ImportDeclaration) return;
        ImportDeclaration importDeclaration = new ImportDeclaration(node);
        node.getChildrens().forEach(a -> {
            if (a instanceof Statement b && b.getChildrens().contains(IMPORT)) {
                importDeclaration.getChildrens().add(a);
                importDeclaration.remove(IMPORT);
            }
        });

        if (importDeclaration.getChildrens().isEmpty()) return;
        node.replace(importDeclaration.getChildrens().stream().findFirst().get(), importDeclaration);
        node.getChildrens().removeAll(importDeclaration.getChildrens());
    }
}
