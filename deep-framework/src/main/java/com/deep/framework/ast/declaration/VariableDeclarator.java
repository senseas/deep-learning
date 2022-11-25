package com.deep.framework.ast.declaration;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.expression.AssignExpression;
import com.deep.framework.ast.expression.Expression;
import com.deep.framework.ast.expression.Name;
import com.deep.framework.ast.type.Type;
import lombok.Data;

import java.util.List;
import java.util.Objects;

@Data
public class VariableDeclarator extends Declaration {
    private Type type;
    private Name name;
    private Expression initializer;

    public VariableDeclarator(Node prarent) {
        super(prarent);
    }

    public static void parser(Node node) {
        if (node instanceof FieldDeclaration) return;
        List<Node> nodes = node.getChildrens().stream().filter(a -> Field_Modifiers.contains(a.getTokenType())).toList();
        node.getChildrens().removeAll(nodes);

        Stream.of(node.getChildrens()).reduce((list, a, b) -> {
            if (a instanceof Name && Objects.nonNull(b) && b instanceof Name) {
                VariableDeclarator declare = new VariableDeclarator(node);
                Type type = new Type((Name) a);
                declare.setType(type);
                declare.setName((Name) b);
                node.replaceAndRemove(a, declare, b);
                list.remove(b);
            } else if (a instanceof Name && Objects.nonNull(b) && b instanceof AssignExpression d) {
                VariableDeclarator declare = new VariableDeclarator(node);
                Type type = new Type((Name) a);
                declare.setType(type);
                declare.setName(d.getTarget());
                declare.setInitializer(d.getValue());
                node.replaceAndRemove(a, declare, b);
                list.remove(b);
            }
            /*if (a.equals(INT)) {
                return;
            } else if (a.equals(LONG)) {
                return;
            } else if (a.equals(FLOAT)) {
                return;
            } else if (a.equals(DOUBLE)) {
                return;
            } else if (a.equals(BYTE)) {
                return;
            } else if (a.equals(CHAR)) {
                return;
            } else if (a.equals(BOOLEAN)) {
                return;
            }*/
        });
    }

}
