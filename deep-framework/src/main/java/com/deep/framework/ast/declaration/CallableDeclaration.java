package com.deep.framework.ast.declaration;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.expression.Name;
import com.deep.framework.ast.expression.ParametersExpression;

import java.util.List;
public class CallableDeclaration extends Declaration {
    private ParametersExpression parameters;
    private Name name;
    private static CallableDeclaration declaration;

    public CallableDeclaration(Node prarent) {
        super(prarent);
    }

    public static void parser(Node node) {
        Stream.of(node.getChildrens()).reduce((List list, Object m, Object n) -> {
            if (m instanceof Name a) {
                if (n instanceof ParametersExpression) {
                    declaration = new CallableDeclaration(node);
                    declaration.setName(a);
                    declaration.setChildrens(a.getChildrens());
                    declaration.getChildrens().add(n);
                    node.replaceAndRemove(a, declaration, n);
                    list.remove(n);
                }
            }
        });
    }

    public void setName(Name name) {
        this.name = name;
    }
}
